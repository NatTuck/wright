# Vision tower

**Status: image support works** - in the engine and through a served preset. The accuracy set
is 19/19 against HF's 19/19 (it was 2/19), the device tower runs on the card by default, and
`qwen3_5-gaudi2` serves images. The per-case table is in "Verdict", the three bugs that were
behind it are in "What it actually was", and the rest of this document is the investigation as
it happened, with the claims that did not survive struck in place.

The Qwen3.8-27B (Qwen3-VL) vision tower, which turns image patches into the
`out_hidden_size` rows that replace a prompt's image tokens.

| file | what it is |
| --- | --- |
| `models/gaudi/resin_models_gaudi/qwen3_5/vision.py` | `VisionTower`, the torch CPU implementation. The **oracle**. |
| `models/gaudi/resin_models_gaudi/qwen3_5/vision_graph.py` | the same tower as one Synapse graph, on the card |
| `models/gaudi/resin_models_gaudi/qwen3_5/vision_device.py` | `DeviceTower`, the card implementation. Same `forward` contract as the oracle. |
| `models/gaudi/resin_models_gaudi/qwen3_5/media.py` | `prepare_image_prompt`, and the switch that picks the tower |
| `models/gaudi/tools/vision_reference.py` | the HF tower as the reference, for diffing `VisionTower` against, stage by stage |
| `models/gaudi/tools/vision_preprocess.py` | `pv` against HF's image processor, one case per resize branch |
| `models/gaudi/tools/vision_accuracy_cases.py` | the 19-case accuracy set, shared by the two runners below |
| `models/gaudi/tools/vision_acc_ours.py` | the accuracy set through the engine |
| `models/gaudi/tools/vision_acc_hf.py` | the accuracy set through HF, on the same ids and pixels |
| `models/gaudi/tools/kernel_check.py` | numeric check of a single custom kernel, without compiling the model |

The harnesses are in `models/gaudi/tools/` with the rest of the model-specific development
tools, not in the package: nothing imports them, and they take a checkpoint path a serving
process never has.

The one-off instruments written while chasing the failure have been removed: the magnitude
sweeps, the span/order probes, the margin and ablation probes, the failed KV and embed
readbacks, and the text-candidate and long-text controls. Their results, and the reason each
was inconclusive, are in "What it actually was" below; the scripts themselves are in git
history. Keeping them in the package implied they were runnable checks when most of them
tested hypotheses that are now settled or void.

The same argument retired five more once the tower was settled: `vision_check.py` (the
per-stage device diff), `vision_e2e.py` (both towers end to end, plus timings),
`vision_decode_positions.py` (the 48-token continuation), `vision_rope_binding.py` (whose four
assertions are now `models/gaudi/tests/test_qwen_vision_rope_binding.py`) and
`splice_probe.py`. Each answered a question that is now closed, and each capability is either
under pytest or covered end to end by the accuracy set.

Two of the checks described here are also assertions under pytest, so they are not only
reproducible by hand: `models/gaudi/tests/test_qwen_vision_rope_binding.py` (card-gated, on
`RESIN_GAUDI_QWEN35`) is the four rope-binding assertions, and
`models/gaudi/tests/test_qwen_vision_preprocess.py` (CPU, gated on the checkpoint) locks the
preprocessing the harness compares against HF. The preprocessing harness stays because the HF
side needs transformers in a separate interpreter, which pytest cannot do; the reference and
accuracy harnesses stay because they are the only things that run the *other* implementation.

## The oracle was never validated

**The CPU tower was never diffed against the reference implementation. It was used as an
oracle for itself for the whole project.** `vision_graph.py` was checked against `vision.py`
and reported agreement to 2.9e-06, which only ever meant "the card agrees with this file" -
it said nothing about whether the file was right. It was not: three bugs, found by diffing
`VisionTower` against HF's `Qwen3VLVisionModel` (`vision_reference.py`), and all three were
invisible to every check the project had.

1. **No vision rotary embedding.** The reference applies axial 2D RoPE to q and k inside every
   vision attention block (frequencies over `head_dim // 4`, h and w angles concatenated and
   duplicated). `vision.py` had no rope at all, so its attention was position-agnostic.
2. **The position table was interpolated with `align_corners=False`.** The reference uses
   `True`.
3. **Patches are consumed in spatial-merge-block order, not raster order.** A `merge x merge`
   block's patches are consecutive in the sequence - the image processor emits them that way -
   so patch 1 of a 2x2-merge grid is at (row 0, col 2). The tower assumed raster order, which
   gave every patch a neighbour's position embedding, rotated it by the wrong angle, and
   merged the wrong four patches. This is the one that broke the image.

After the fixes `VisionTower` matches the reference on a 16x16 patch grid at depth 27:

| stage | relative error |
| --- | --- |
| patch embed + position | 6.8e-07 |
| after all 27 blocks | 2.0e-05 |
| merger output | 1.5e-04 |

The lesson worth carrying further than these three bugs: a reference you wrote is not a
reference until something independent has agreed with it. Every "verified" number in this
document before this section was device-versus-its-own-reference.

### The device graph now matches the reference too

The device disagreed with the corrected oracle by 2.99e-01 at attention, and the cause was not
the rope: `vision_device.position_bias` called `interpolate_positions` directly and never
applied the merge-order gather that `VisionTower._pos_embed` does, so the graph's position
input was in raster order while everything else was in block order. The `embed` check missed
it because its reference used the same wrong array - a second reference going stale in exactly
the way this document warns about. One line to gather in merge order fixed every downstream
stage.

Final numbers, all at depth 27:

| comparison | relative |
| --- | --- |
| CPU tower vs HF, patch embed + position | 6.8e-07 |
| CPU tower vs HF, after all 27 blocks | 2.0e-05 |
| CPU tower vs HF, merger output | 1.5e-04 |
| **device vs HF, merger output** | **1.6e-04** |
| device vs CPU tower, 64 patches | 9.7e-07 |
| device vs CPU tower, 256 patches | 6.3e-06 |
| device vs CPU tower, block 0 | 7.6e-07 |

`vision_reference.py --device` is the direct card-versus-HF comparison; the device path also
exercises the rope kernel, the merge kernel and the output readback end to end.

### The device pieces that were needed

- `rg_rope_vision_f32`: axial rope on q and k, per-token cos/sin table. The head dim is padded
  into a permuted layout (first half in lanes [0,36), second half in lanes [64,100)) so the
  half-swap is a 64-lane aligned block swap; q and k share the permutation, so q.k is
  unchanged, and zero lanes pair with zero lanes, so the padding stays zero. Verified in
  `rope_probe.py` (since removed, in git history): kernel vs numpy 0.0, q.k preserved 0.0,
  padding zero.
- `rg_merge4_f32`: four consecutive columns into one row - a strided copy on the card even
  though it is a `view` on the host, because feature-major means memory index is
  `feature + H*token`.
- `VISION_ROPE_IDENTITY=1` in the harness makes the graph's rope an identity so the unroped
  sub-stage references stay comparable.

## Verdict: image support works, in the engine and through the server

**The engine answers 19 of 19 image questions correctly, and so does a real server with the
preset flipped. HF answers 19 of 19.**

The set (`vision_accuracy_cases.py`) is 8 solid colours, 8 split-colour mirror pairs and 3
shapes, one unambiguous prompt each. Per case, the engine (thinking disabled, greedy) and the
server (module 6, port 8011, thinking on, 400 tokens of room):

| case | expected | before the fix | engine now | server now |
| --- | --- | --- | --- | --- |
| solid-red | red | 'blue' | Red | Red |
| solid-blue | blue | 'Blue' | Blue | Blue |
| solid-green | green | 'Blue' | Green | Green |
| solid-yellow | yellow | 'Blue' | Yellow | Yellow |
| solid-black | black | 'White' | Black | Black |
| solid-white | white | 'Blue' | White | White |
| solid-orange | orange | 'Blue' | Orange | Orange |
| solid-purple | purple | 'White' | Purple | Purple |
| split-red-blue | red | 'blue' | red | Red |
| split-blue-red | blue | 'red' | blue | Blue |
| split-green-red | green | 'blue' | green | Green |
| split-red-green | red | 'blue' | red | Red |
| split-blue-green | blue | 'red' | blue | Blue |
| split-green-blue | green | 'green' | green | Green |
| split-yellow-blue | yellow | 'blue' | yellow | Yellow |
| split-black-white | black | 'red' | black | Black |
| shape-circle | circle | 'triangle' | Circle | Circle |
| shape-square | square | 'Triangle' | square | Square |
| shape-triangle | triangle | 'Triangle' | Triangle | Triangle |

The "before" column is the recorded 2/19 run: the two it got were `split-green-blue` and
`shape-triangle`, and the rest were wrong and unrelated to the image. Every case flipped
independently - the solid colours, both orders of every mirror pair, and all three shapes - so
nothing here is carried by one easy case.

The server's median was 0.8 s per request in the run that produced this table and 1.3 s in the
last run, and the difference is the configuration rather than the machine: the later run is the
shipped one, with MTP on (`draft_tokens=2`), which routes an image sequence to plain decoding and
so gives up speculative drafts for it. A timing here is only meaningful with that named.

Note what the two columns say together: the split pairs are the discriminating cases, because
they require the model to know *which side* a colour is on, and a permuted image cannot answer
them. Before the fix one of the eight was right; the chance of that being luck is not worth
arguing about, and after the fix all eight are right in both orders.

## Decode positions after an image

A second, smaller bug fixed in the same pass, and the one the accuracy set above cannot see: it
scores the first token, and the first token is sampled by the prefill.

An image prompt's 64 image tokens occupy only 8 positions (`max(grid_h, grid_w) / merge`), so the
prompt's compressed positions fall behind its token count by
`delta = max(position_ids) + 1 - tokens = merged - image_tokens` (-56 here). HF adds that delta to
every later position (`mrope_position_deltas`). We advanced the decode position by token count, so
every step after an image rotated at the token index.

The comparison that can see it is a long greedy continuation, ours against HF, token for token:

| case (48 tokens after an image) | with the delta | with the token index |
| --- | --- | --- |
| describe-split | 47 / 48 | 13 / 48 |
| describe-shape | 48 / 48 | 48 / 48 |
| story-split | 47 / 48 | 2 / 48 |

Our delta equals HF's `rope_deltas` exactly on all three. `describe-shape` is robust both ways -
a circle's description is predictable - which is why the ablation matters: any single case would
have shown nothing. The 48-token continuation was the card-side regression; it also asserted that
a decode step's logits *move* when the offset changes (0.89 / 1.50 / 0.64), so the offset cannot
be a silent no-op.

## What it actually was: the splice uploaded the features transposed

One bug, and it is a layout convention, which is why every magnitude-based check in this
document passed over it for the whole project.

`splice` declares its `img` input as a logical `(H, T)` tensor and reads `img[h, t - first]`.
Device tensors are **token-major** - a logical `(n, T)` tensor lives at flat `t * n + i` - so the
host array for a `(H, T)` tensor is `(T, H)`, with the features in its first `count` rows. The
caller uploaded the compact `(H, count)` features instead, so feature `k` landed at flat
`h * count` where the kernel read some other `(h, t)`.

The consequence is the exact shape of the symptom this document spent hundreds of lines on: the
image columns held **the right values in the wrong places**. The value multiset was preserved
exactly - the maximum came out at 85.386 both ways - so nothing that measured magnitude could
see it. The model could tell an image was present and never what it was.

| check | before | after |
| --- | --- | --- |
| our embed's image columns vs the tower's features | 85.7488, permuted | **0.0000, exact** |
| accuracy set | 2 / 19 | **19 / 19** |

It also retro-explains three findings that had been read as deep properties of the engine:

- **"gain/scale is noise at every scale, no dose response"** - a permuted image has no dose
  response at any scale. Nothing was saturating or attenuating; the content was never there.
- **"HF's own features attenuate identically in our engine"** - of course they did. Any features
  put through the same upload arrive scrambled the same way, so the result could not depend on
  which implementation produced them.
- **"HF with our features answers correctly"** - the features were always fine. Only their
  placement was wrong.

The second real bug found in the same pass, fixed and separately worth having: a media plan bound
`cs_tab` to the *text* rope table's buffer, uploaded the mrope rows over it behind the launches,
and restored it afterwards, so the launches read whatever the text table was. Two materially
different row sets gave bit-identical output - which is what "our engine ignores the rope it is
given" was measuring. A media plan now has its own buffer, uploaded blocking, bound as an appended
binding on the pre launch; three row sets (mrope, zeros, the standard table) give 5.88 / 1.65 /
4.68 max logit difference. `models/gaudi/tests/test_qwen_vision_rope_binding.py` is that assertion under pytest.

Fixing the rope alone did **not** fix the answers (2/19 before and after, with the wrong words
shifted slightly). Fixing the splice did. Both were real; only one was the cause.

### Everything below this line is diagnosis, and some of it is now known to be wrong

The sections that follow were written while the splice bug was live, and several of their
conclusions are struck by it or by a bad instrument. They are kept rather than deleted, with the
corrections marked, because the way they failed is the useful part.

The instrument worth naming first: **every "our hidden state vs HF" comparison in this document
read the carry as `d2h((H, T))`.** The convention is `d2h((T, H)).T`. Checked against the
checkpoint's own `embed_tokens` rows, the correct read matches to 0.000000 and the transposed one
correlates at 0.0026. The text columns of a correct read come out at std 0.2968 against HF's
0.2985 after one layer; the transposed read reported 1.598 relative error on the text prefix,
which should have been the tell - a *text* prefix that far from HF is impossible in an engine
whose text generation tracks HF.

### What the diagnosis established

The margin (`logit(red) - logit(blue)`, thinking disabled, identical ids and pixels on both
paths) is useful for locating the loss and useless as a gate. On text it tracks HF closely at
every magnitude - seven prompts from margin 0.2 to 12, ratios 0.92 to 1.25, including
`bigger` 0.634 vs 0.609 and `cold` 0.839 vs 0.914 - so the engine's arithmetic does not lose
small margins. On images it loses 90%: split-A 0.086 against HF's 5.375.

Per-layer, truncating both sides to N layers and comparing the next-token margin:

| N | HF | ours |
| --- | --- | --- |
| 1 | +0.708 | +0.727 |
| 8 | +0.220 | +0.122 |
| 16 | -0.280 | -0.518 |
| 32 | -1.708 | -1.514 |
| 64 | **+4.031** | **+0.388** |

Text tracks HF at all 64 layers and on a prompt longer than the image prompts (84 tokens:
+3.572 vs +3.938). The image margin tracks through layer 32 and collapses by 64. The relative
logit difference accumulates smoothly from 6.9e-03 at N=1 to 2.2e-01 at N=64, so the loss is
not at one layer.

**VOID: the two sentences above.** This table came from a logit lens that applied the final norm
it should not have applied, and which failed its own calibration (-0.50 against a plain forward's
-0.19). "Tracks to layer 32, collapses by 64" was never established, and neither was "not at one
layer": the loss was at the splice, before layer 1. The lens's text agreement was the tell - a
text margin that tracks HF is what a correct lens would show too, so it could not distinguish a
working instrument from a broken one.

A short span was tested explicitly, because every image prompt until now straddled the
`r=64` page/sigma boundary while every text control was short. Lowering the recipe's
`min_pixels` to 1024 keeps a 64x64 image at 64x64, giving a 4x4 patch grid and a **four-token**
span at 15..18, entirely inside the first page - with the recipe's own prompt and thinking off.
It fails exactly like the 64-token span:

| image | span | answer | expected |
| --- | --- | --- | --- |
| red | 15..18 | 'Blue' | red |
| blue | 15..18 | 'black' | blue |
| green | 15..18 | 'Blue' | green |
| red | 15..78 | 'blue' | red |
| blue | 15..78 | 'White' | blue |
| green | 15..78 | 'Blue' | green |

So the boundary is exonerated, and with it the whole "multi-block span" family of explanations:
a four-token span that never leaves the first page is as wrong as a sixty-four-token one. The
problem is in how the image rows are used generally, not in how a span is mapped across pages.

**Cross-feed, direction 1: our features are good and HF uses them correctly.** Rather than
infer from eliminations, the same spliced rows were fed to HF and to our engine. Building the
embedding on the host - the token embedding with our 64 vision rows written into the span, same
ids, same framing, same `image_grid_thw` and `mm_token_type_ids` - and generating from
`inputs_embeds`:

| rows spliced into HF | answer |
| --- | --- |
| **our features** | "The color on the **left half** of the image is **red**" |
| HF's own features | the same |
| all zeros | "red" as well |

So HF, given our features, answers the question correctly - the features are right and the tower
comparison was not misleading. The same features through our engine give 'blue' on this prompt.
**That is the direct statement that the defect is in the engine, not in the features**, and it no
longer rests on elimination.

The single-case caveat was resolved with the full four-way table - red and blue solids, our
features and a zeros control, through both engines, same ids and framing throughout:

| image | HF + our features | HF + zeros | ours + our features | ours + zeros |
| --- | --- | --- | --- | --- |
| red | **'Red'** | 'Blue' | 'green' | 'Blue' |
| blue | **'Blue'** | 'Blue' | 'Blue' | 'Blue' |

HF with our features is **discriminative and correct**, and the zeros control breaks it - red
becomes 'Blue' - so the case is valid and our features carry the information. Our engine is not
discriminative at all: red gives 'green', blue gives 'Blue', and the zeros control gives 'Blue',
i.e. it answers 'Blue'-ish regardless of the image.

That is the airtight form of the conclusion: **the features are good (HF proves it on a
discriminative case), and the engine does not use them.**

How much of them it loses, and whether it depends on the values, is now measured. Feeding the
*same* rows to both engines, the margin at the last token:

| rows | HF | ours |
| --- | --- | --- |
| HF's own features | **+5.4375** | **-0.2910** |
| our features | +5.4375 | -0.1629 |
| zeros | - | +0.0179 |
| text only | -0.1875 | -0.1629 |

So HF extracts a margin of 5.44 from features that our engine turns into 0.29 - and it does so
with **HF's own features too**, not just ours. The engine's answer therefore does not depend on
which implementation produced the features: it attenuates their effect by roughly 17x either way.
That rules out feature content as the cause and makes it structural in the engine.

**STRIKE - "our engine ignores the rope rows it is given" (corrected above).** The
observation was right and the inference was wrong. A media plan bound `cs_tab` to the text
table's buffer and uploaded the mrope rows over it *behind* the launches, then restored it, so the
launches genuinely read the text table - a binding bug, not "rope content does not matter". Two
different row sets now give 5.88 / 1.65 / 4.68 max logit difference. The 1.999 "the two rope sets
have nothing in common" is also a malformed reconstruction: HF's `rotary_emb` already returns
`cat((freqs, freqs), -1)`, so rebuilding `[cos | cos | -sin | sin]` from it double-duplicates.
The mrope arithmetic was checked against HF's source and is correct, including the interleaved
`f % 3` stream assignment for sections [11, 11, 10].

**The rope cross-feed: our engine ignores the rope rows it is given.** HF's actual per-token
rows were captured from `model.language_model.rotary_emb` (after `recomposition_frequencies`) and
rebuilt as `[cos | cos | -sin | sin]`. They differ from ours by **1.999** on a range of +/-1, i.e.
the two rope sets have essentially nothing in common. Feeding HF's rows into our engine in place
of ours, same features, same ids, same span, at one layer:

| rope rows | image columns max | last token max |
| --- | --- | --- |
| ours | 215.364 | 6.046 |
| HF's | 215.364 | 6.046 |

image columns differ by **0.000000**, last token by 0.000000, text-before by 0.000000. Completely
different rope content produces a bit-identical result, so the rows passed as `p.rope` are not
reaching the table the graph reads. That is the mechanism the "presence not content" result has
been describing: the 64 image columns are spliced in and attended, but they are rotated as though
they were text, so their content cannot be recovered while their presence still registers.

The rope content itself is therefore unverified as well - the earlier 0.0019 agreement was against
our own reconstruction of HF's rope, not against these tensors, and these tensors disagree with
ours by 2.0.

**STRIKE the conclusion, keep the measurement - "the upload-ordering mechanism was tested and is
wrong".** Determinism across five runs did not show the buffers were ordered; it showed the same
stale table was read every time, because the binding never pointed at the new one. `_upload`'s
split meaning was real and is now fixed (a large table goes as queued chunks, so it means the same
thing at every size), and the media rope has a dedicated buffer so ordering cannot matter. The
question "is `h2d_nb` ordered?" is therefore moot here rather than answered.

**The upload-ordering mechanism was tested and is wrong.** `_upload` does use a blocking `h2d`
above 64 KiB and a queued `h2d_nb` below it, and `_splice_media` mixes both - but the prefill is
bit-identical across five runs in one process (logits differ by 0.0), and replacing all three
media uploads with blocking `h2d` changes nothing. So the image path is not reading a stale or
reordered buffer. Determinism also means the scale sweep's incoherence is a property of the
model's response, not of the measurement.

**VOID - "the first divergence is layer 1, in the image columns".** Both columns of that table
were read with `d2h((H, T))`; the convention is `d2h((T, H)).T`. Re-measured correctly, our text
hidden-state std after one layer is 0.2968 against HF's 0.2985, and there is no layer-1 divergence
to localise. The image columns of the *embedding* were wrong - permuted by the splice - which is
what the 219% was picking up, at the one place a correct read would have found it immediately.

**The first divergence is layer 1, in the image columns.** One layer, red versus blue features,
same ids and span:

| | last token | image columns | text columns |
| --- | --- | --- | --- |
| HF, red vs blue | **0.0022** | 31.50 | 0.000 |
| ours vs HF, red | 0.99 relative | **2.19 relative** | - |

HF's last token is *insensitive* to the image content after one layer - it moves by 0.0022 while
its image columns move by 31.5 - which is what a first attention layer that has not yet mixed the
span into the output looks like. Our engine differs from HF by 219% at the image columns after
that same layer. So the content is present at the columns in both, and the two engines disagree
about what those columns *are* by the end of layer 1: this is not a late accumulation, it is the
first layer. That is where the code should be read.

**STRIKE - "the attenuation is structural, not a gain error".** It was not a gain error, which was
the right half. The other half - "structural in the engine" - was wrong: the features arrived
permuted, so every scale of a permuted image is equally uninformative. See the verdict.

**The attenuation is structural, not a gain error.** Scaling the real features across two
orders of magnitude, one engine build, text-only and zeros baselines in the same run:

| features | margin | effect vs text-only |
| --- | --- | --- |
| text only | -0.1629 | - |
| zeros | +0.0179 | +0.1808 |
| real x0.1 | -0.2177 | -0.0548 |
| real x1 | -0.1629 | +0.0000 |
| real x10 | -0.7293 | -0.5663 |
| real x40 | -0.2555 | -0.0925 |
| real x100 | -0.6173 | -0.4544 |

The effect of the rows is noise-level at every scale and does not grow with the signal: it
oscillates between -0.57 and +0.18 with no trend. HF takes the same x1 features to +5.4375. So
this is not a gain or normalisation factor that a corrected multiplier would fix - the image's
influence is structurally absent, and no amount of signal recovers it. The zeros arm (+0.1808)
being as large as any real arm is the same statement from the other side: what the model reacts
to is the *presence* of 64 spliced columns, not their content.

**A void claim, recorded as void.** The earlier "the image margin tracks HF to layer 32 and
collapses by 64" came from the logit lens that applied the final norm it should not have applied,
and which failed its own calibration (-0.50 against a plain forward of -0.19). That number has
been the most load-bearing claim in this document and it is **void**; it has not been re-derived
with the calibrated lens. Every number in this section now states which instrument produced it.

Two further probes closed the obvious follow-ups:

- **magnitude is not the variable.** One-hot rows at max 10, 30, 90 and 300 give margins of
  -0.2472, -0.4131, -0.4867 and -0.1518 - flat, no dose-response - so it is not saturation or a
  scale threshold.
- **the one-hot probe is not discriminative**, contrary to the hope: HF gives it +0.0000 and ours
  -0.2472, i.e. both small. A one-hot row is not a real feature and neither engine reacts to it,
  so the structural-versus-numeric question had to be settled by the cross-feed instead.

The logit lens was also calibrated before being believed: `head(hidden_states[-1])` with **no**
norm reproduces the plain forward exactly (max diff 0.0000), because `hidden_states[-1]` is
already post-norm. The first attempt applied the norm and failed its own calibration, and every
number from it was discarded.

**The preprocessing is verified, and it was the last link that had never been compared.** The
accuracy set compares HF through `AutoProcessor` on a PIL image against our Rust recipe - two
different preprocessors - and every earlier tower-versus-HF check fed HF *our* `pixel_values`,
so the comparison validated the tower on our pixels without ever testing whether our pixels were
right. For a solid-red 64x64 image:

| | shape | grid | per-channel mean (first patch) |
| --- | --- | --- | --- |
| our recipe | (256, 1536) | (1, 16, 16) | [0.0, 0.0, -1.0] |
| HF image_processor | (256, 1536) | (1, 16, 16) | [0.0, 0.0, -1.0] |

**max abs difference 0.000000** (relative 2.4e-07, float rounding). A channel permutation would
show as 2.0 and shows as 0.0, so there is no RGB/BGR swap; the resize, the normalisation, the
patch layout and the grid all agree exactly. The channel-swap explanation - which fitted
red -> "Blue" better than anything else - is dead.

Excluded, each by measurement rather than argument: the tower (1.6e-04 against HF, and HF's
own merger output has the same 91.45 max so the 47x scale over the text embeddings is by
design); the splice (64 of 64 image columns differ from the text-only embedding, exactly the
span, all equal to the features); the mrope rows (cos and sin against the rope HF actually
applies, 0.0019 over the image span); the prompt (framing tokens at 56 and 121, positions
continuous across the span); fp8 precision (the text control above); the span's position in
the sequence (image-first and image-later fail identically); the sigma row-permutation boundary
at r=64 and sequence length (the 84-token text prompt crosses it and tracks HF); and "the rows
are ignored" (ablating the features to zero moves the margin toward the no-image baseline,
+0.161 vs +0.173, while the real features give +0.072).

### What was left, at that point

**Historical. The premise of this section is VOID and its two leads were never needed.** "The
image margin tracks HF to layer 32 and collapses by 64" is the logit-lens reading struck
elsewhere: the lens applied the final norm it should not have and failed its own calibration.
And the loss was never in the later layers - the features arrived permuted at the splice, which
is upstream of every layer. Kept because the KV instrument's seven failures are described below
and are worth reading.

At the time, the image margin appeared to track HF to layer 32 and collapse by 64, with every
input verified, so the loss looked specific to the image rows and late. Two structures in that
path had not been inspected:

1. **the K/V actually stored for the image span** - `kv_rows` maps a position to an arena row,
   K rows are natural order and V rows are `sigma_inv`-permuted within 128-token blocks, and
   `ROW_BYTES` is 512 (256 bf16, two groups of 128). The span 57..120 straddles r=64, so
   positions 57-63 and 64-120 land in different halves of the permutation. The long-text
   control crosses that boundary with *text* rows, which is not the same test;
2. **the image positions' hidden state at layers 32 and 64** - cheaper, but the per-layer
   readback must first be validated against the final hidden state, since an earlier attempt
   disagreed with HF on text too and was therefore measuring the wrong thing.

Start with (1): dump K and V for positions 58, 60, 63, 64, 66, 70 and compare each against
HF's `past_key_values` at the same positions. If the first half matches and the second does
not, the sigma boundary applied to the image span is the bug. If all six match, the storage is
right and the defect is on the read side or in the layers, and (2) is next.



Measured as `logit(red) - logit(blue)` at the first generated token, thinking disabled, with
both paths fed the *same* ids, pixels and grid:

| case | HF | ours | HF argmax | ours argmax |
| --- | --- | --- | --- | --- |
| split-A (red\|blue) | +5.375 | +0.086 | ' red' | ' green' |
| split-B (blue\|red) | -5.312 | -0.026 | ' blue' | ' Blue' |
| solid-red | +6.500 | +0.629 | ' red' | ' red' |
| solid-blue | -6.000 | +0.637 | ' blue' | ' red' |

HF is large and correct on all four. Ours has the right sign on three and a magnitude 10-100x
smaller, so the argmax is decided by noise. That is a real defect, not a probe that is too
sharp.

Ruled out, each by direct measurement rather than reasoning:

- **the tower**: 1.6e-04 against HF on a real image, and HF's own merger output has the same
  max (91.45 vs 91.43) so the 47x scale difference from the text embeddings is by design;
- **the splice**: 64 of 64 image columns differ from the text-only embedding, they are exactly
  the span, and every one equals the tower's features;
- **the mrope rows**: compared against the rope HF itself applies, captured with a forward hook
  on `model.language_model.rotary_emb` - cos and sin agree to 0.0019 over the image span (and
  the same on the text prefix), which is bf16-versus-f32 rounding;
- **the preprocessing**: bit-identical to HF's image processor on the same image (max abs
  difference 0.000000, no channel permutation);
- **the prompt**: the framing tokens are present (`<|vision_start|>` at 56, `<|vision_end|>` at
  121) and the positions are correct and continuous across the span.

So every input to the LLM is verified and the compression is downstream of them. The next cut
is the hidden state after layer 0 for an image token, against HF's `output_hidden_states`. A
first attempt read `engine.carry[1]["x"]` and got uninitialised memory (1e33) - the carry is
sized for the largest mode, so the read needs the prefill bucket's column count - and that is
where this stands.

## The KV layout, and why the dump is not yet trustworthy

Derived from `cache.py`, `attention.py:kv_rows`, `model.py:_bind_table` and the header of
`kernels/kv_store.c`, for this checkpoint (hidden 5120, heads 24, head_dim **256**, kv_groups 4,
16 full-attention layers of 64, `PAGE_TOKENS` 128, `ROW_BYTES` 512 = **256 bf16**):

- one arena row holds **one group's head_dim values for one token** for K *or* V;
- `kv_layer_rows = 2 * kv_groups * PAGE_TOKENS = 1024` rows per full-attention layer, and a
  layer's region starts at `arena + a * kv_layer_rows * ROW_BYTES` where `a` is the layer's
  index *within the full-attention set* (`self.full.index(layer)`), not the absolute layer;
- the destination row for token `t`, group `g` is `kv_rows[t, g]` for K and `kv_rows[t, 4+g]`
  for V, and those entries already include the page base, so the byte address is
  `arena + (a * kv_layer_rows + kv_rows[t, col+g]) * ROW_BYTES`;
- K rows are stored in natural token order; V staging rows are `sigma_inv`-permuted within
  128-token blocks, so token `t`'s V is at page row `sigma_inv(t % 128)`. Read back from the
  device, `kv_rows` shows this exactly as expected: for the first token the K/V rows are
  `base+0`/`base+0`, for `t=1` `base+1`/`base+2`, for `t=63` `base+63`/`base+126`, and for
  `t=64` `base+64`/**`base+1`** - the permutation switching halves at r=64, confirmed.

**The dump itself does not yet validate.** Reading K and V this way gives values of plausible
magnitude, but comparing them against HF's `past_key_values` at the same positions is ~100%
wrong *at text positions* (0, 1, 2) as well as at the image positions (58, 60, 63, 64, 66, 70).
Since text answers are correct, our cache cannot be wrong for text, so the interpretation is
what is wrong - and the first version of it was worse still: it read the rows as f32 when they
are 256 bf16, which the text control caught immediately.

Two candidates, neither confirmed:

1. **K is not comparable at all.** `attn_prep` applies the Q/K norm and rope *before*
   `kv_store` writes, so our stored K is post-rope while HF's cache is pre-rope. A direct
   comparison of K is therefore invalid by construction and needs the rotation undone first.
2. **V should be comparable and is not**, which means something else in the address or the row
   layout is still wrong. The most likely remaining error is the row's interpretation (one
   group per row, group-major) or the arena base.

A third attempt, this one from our own engine and so needing no cross-implementation mapping:
a single-token prefill with `layers=4`, then compare the stored V row for token 0 against the v
part of `p_out` - the fused qkv projection that `attn_pre_graph` writes before `attn_prep`
splits it, and which is the exact quantity `kv_store` should have stored (v has no rope and no
norm). It does not match either: the stored values are around 0.7 where the projection is 10 to
23, a factor of ~20, consistent across all four groups. So either the arena base or the layer
offset in this note is wrong, or there is a scale factor between `p_out` and what is stored that
the note does not account for. Either way the note's addressing is **not validated**.

A fourth attempt found the scale's *mechanism* but not its value. `attn_prep.c` line 12 says
it plainly: "FP8 builds (-DFP8): p is an FP8-GEMM output with per-token scale ps (1,T) (input
1)", applied at `TOKEN_SCALE(ps, t, ...)`. So `p_out` is the raw FP8 accumulator and the real
projection is `p_out * ps[t]` with `ps` = the carry's per-token `hs`. But the engine's
`hs[0]` for that layer is 0.371, not the ~20 the ratio needs, and applying it moves the
comparison further away (stored ~0.7, `p_out * hs` 2.2-3.8 per element, raw 10-23). So the
missing multiplier is real and documented in the kernel, but it is not `hs` as read here -
either the layer whose `hs` I read is not the one whose `p_out` I read, or a second scale
enters between the projection and the store.

The per-element ratio settles which kind of error it is: comparing the stored V against
`p_out[13312:14336] * hs[0]` (the v part, the parent's corrected offset) and against
`p_out[12288:13312] * hs[0]` (the k part) both give a **non-constant** ratio - medians 0.004 and
0.003 with spreads of 1.05 - so this is not a scale factor, it is the wrong row or the wrong
base address. The specific clue: `kv_rows[0, 0]` is exactly **311296**, which is exactly
`CacheGeometry.slab_rows` for this checkpoint. That is too exact to be a coincidence, so the
row numbers in `kv_rows` are most likely slab-relative (or offset past the state block) and the
base this note uses - `engine.arena + row * ROW_BYTES` - is missing that offset.

So the next read is: find how a `kv_rows` entry is meant to be turned into an arena address
(`_pages`/`plan_slots`, or the arena view `(256, PR, NB)` in `kv_store.c`), which is one
function, and the note's last unvalidated step disappears.

The suggested decomposition - `arena[:, row % PR, row // PR]` instead of the flat row - was
tested and is **wrong**: it reads zeros (unwritten memory) where the flat index reads written
data. `kv_store.c`'s own comment agrees with flat: "the arena is viewed as (256, PR, NB) bf16:
row r is (., r % PR, r / PR)", and with dim 0 contiguous the element offset of `(r % PR,
r / PR)` is exactly `r`. So `arena + row * ROW_BYTES` stands.

The stored row then matches **nothing** in the projection: searching every p_out column, group
and of the first four tokens for the best cosine similarity against the stored V row of token 0
gives |cos| = 0.084 - no correspondence at all, at magnitudes ~1.5 against the projection's
10-23. So the row being read is not this layer's v for this token, which points at `p_out` not
holding what the note assumes (a different layer's projection, or the buffer being reused) or
at the layer offset, rather than at the addressing.

A seventh attempt tested the `pack_sigma` hypothesis - that `attn_prep.c:51-53` stores V
interleaved across the two 64-lane halves (`out[2i] = v[i], out[2i+1] = v[64+i]`) - by building
the expected row as `pack_sigma(projection * hs)` per 128-lane half. It also fails: cosine
similarity against the stored row is -0.005, +0.020, +0.107, -0.035 across the four groups.

**Recommendation: abandon this instrument rather than continue it.** Seven attempts have each
been killed by their own check - f32-versus-bf16, K being post-rope, a ~20x magnitude gap, the
`hs` scale not fitting, a wrong-row/offset verdict from the per-element ratio, the flat-versus-
decomposed addressing (flat is right, the decomposition reads zeros), and now the sigma packing
of the values. The instrument has cost seven runs and has not validated once. Nothing in the
product verdict depends on it. At the time, the accuracy set stood at two correct answers of
nineteen against HF's nineteen, and the per-layer margin sweep appeared to track to layer 32 and
collapse by 64, and those two were taken to say what was known and what was not. Both readings have since changed: the
accuracy set is 19/19, and **the margin sweep is VOID** - it came from a logit lens that applied
the final norm it should not have and failed its own calibration (-0.50 against a plain forward's
-0.19), so nothing was ever established about layers 32 and 64. The instrument's abandonment
stands on its own seven failures, which is all this paragraph needs to say.

The next instrument was to be the embedding comparison - HF's `hidden_states[0]` against our
spliced `x0` - which needs no layer readback and no carry indexing. It could not be set up: a
hand-built binding of the media embed graph (`E_gather`, `tok`, `next_norm`, `img`, `img_span`
plus the four carry outputs, sized from `carry_out`) launches with status 30, "Sections'
validation failure", with all nine slots confirmed present. So the sizes or the layout of my
reconstruction are wrong somewhere.

That is the same failure as the KV dump in a different place, and it has the same answer: **do
not reconstruct the engine's bindings by hand.** The engine already holds them -
`engine.binds[("embed_img", mode, -1, "")]` is the exact slot/address table it launches with, and
`_bind_table` is the code that builds it. Reading `x0` out of a real prefill through those, or
dumping the bound addresses and reading the buffer they point at, removes the reconstruction
entirely. Every one of the failed attempts this round, the seven KV ones and this one, failed in
a reconstruction rather than in the measurement.

If someone does continue it, the right method is not another hypothesis: read
`attn_prep.c` end to end for the V path - what `pack_sigma`'s inputs are, whether `hs` is applied
before or after it, and whether `vn` is the buffer `kv_store` reads - and then compare against
`vn` itself (a graph tensor that can be bound and read) rather than against a value
reconstructed from `p_out`. Comparing against the kernel's own output removes every assumption
at once, which is what the last seven attempts each got wrong in a different way.

**The cache instrument is therefore unvalidated, and further attempts are not recommended**: the addressing, the row layout and
the sigma permutation are all confirmed in the data, and the one thing missing is a single
multiplier whose mechanism is named in the kernel header and whose value is not yet located.
That is a two-step gap for the next person, not a wall: find the scale between `p_out` and the
stored row, re-validate V on text, and then the image V dump at 58/60/63/64/66/70 becomes
usable.

Until that is settled, **no conclusion about the image span's K/V may be drawn** - the
numbers above say only that the instrument is unvalidated, which is exactly what the text
control exists to detect. The next person should compare against a reference derived from *our*
hidden states rather than HF's cache, or undo the rope on our K first.



Within a process the engine is exactly reproducible - four repeats of the same prefill, image
and text, give a 0.0 difference. Across processes it is not: the same image through the same
tower gives features that differ by up to 0.024 on a scale of 88.5, i.e. **relative 2.7e-04**,
spread over every row and column rather than localised. It is not the recipe cache (a fresh
cache directory reproduces the variation) and not an unwritten buffer (that would give garbage,
not 1e-4); the most likely cause is GEMM accumulation order moving with the process's workspace
layout.

That matters because it is amplified. On the colour prompts the answer is marginal - margins
around 0.1 against HF's 5.4 - so a 2.7e-04 input difference moves the reported margin between
roughly 0.07 and 0.20 for the same input. If you compare our numbers across runs and see that
spread, it is this, not a regression.

The assertable properties are therefore: **within a process, bit-identical**; **across
processes, within ~1e-4 on features**, with the resulting logit spread stated rather than
assumed to be zero. A bit-identical-across-processes invariant cannot hold on this stack.

## Versions

The reference numbers in this document were produced with **transformers 5.22.1**, which is
what `.venv-ref` pins. The repository pins **5.14.1** in `uv.lock`. The two are not
interchangeable and the numbers are only comparable within a version - the same discipline as
voiding the per-stage numbers that predated the patch-order fix. The oracle deliberately lives
in its own venv (`.venv-ref`) so the reference never shares an interpreter with the thing it
judges; installing torchvision into the main venv once upgraded torch from the pinned 2.13.0
to 2.15.0 and that is why the split exists.

## Process lessons

**The question before trusting a check is not "did it pass" but "could it have failed".**

Every lesson below is that question, asked too late. A check that cannot fail is worse than no
check: it produces a green line and a settled belief. The pattern is always the same - the input
or the instrument could not distinguish the hypothesis, so it agreed.

1. **A reference you wrote is not a reference until something independent agrees with it.** The
   CPU tower was an oracle for itself for the whole project and had three bugs.
2. **A stage that is green while its consumer is red is a reference bug, not a device bug.**
   `embed` passed at 2.5e-07 against the same wrong position array the graph was fed, while
   `ln1` was 4.13e-01.
3. **The thing you assert with has to be checked too.** The colour acceptance test scored 8-10
   tokens, which for a reasoning model is only the preamble; HF fails identically at that
   budget.
4. **Accepting a hypothesis and testing it with a run, when the answer is in a file already
   read.** Two of the KV instrument's seven failures were this: the flat-versus-decomposed
   arena addressing was settled by a comment in `kv_store.c` that had already been quoted in
   this document, and the dtype was settled by `ROW_BYTES` being 512. Both were read, not run.
5. **The same habit in the other direction**: proposing a tensor decomposition from a declared
   shape without reading the kernel that addresses it. The decomposition that was offered read
   zeros - unwritten memory - and one run killed it.
6. **A layout convention is invisible to every check that measures magnitude.** The splice bug
   preserved the value multiset exactly - the maximum was identical to six digits - while
   destroying the image. Twelve probes compared magnitudes and all passed. The one assertion that
   would have caught it is "the image columns *are* the features, element for element", which is
   one `array_equal` and was never written.
7. **An instrument that cannot distinguish the hypothesis looks like agreement.** Four
   separate times: a uniform image cannot detect a mask error or a resize difference; a bf16
   execution path cannot adjudicate an f32 tower (the "2.0e-01 fidelity gap" was HF's bf16
   end-to-end path disagreeing with HF's own f32 tower, not our tower); a magnitude comparison
   cannot detect a permuted image; and a reconstruction of HF's rope cannot verify HF's rope.
   In each case the check ran, passed or failed, and meant nothing. The question to ask before
   trusting a check is not "did it pass" but "could it have failed".
8. **Read the buffer back the way the code writes it.** Every hidden-state comparison here used
   `d2h((H, T))`; the convention is `d2h((T, H)).T`. That single transposition produced a
   "layer-1 divergence" that sent the search to the wrong layer, and it survived because a
   plausible-looking relative error (1.598 on a *text* prefix that demonstrably tracks HF) was
   accepted as a finding instead of as evidence the instrument was broken.

The transferable form: when a measurement and the code disagree, read the code; when a
hypothesis arrives from outside, check it against the source before spending a run on it. Six of
the seven KV attempts were experiments that a five-minute read would have replaced.

## Known failures

Seven incidents, all the same shape: **the thing used to assert was itself unchecked, or
could not have failed.**

1. **The oracle was an oracle for itself.** `vision.py` was diffed against the device and
   reported 2.9e-06 for the whole project without ever being compared to the reference
   implementation. It had three bugs (missing vision rope, `align_corners=False`, and patches
   consumed in raster rather than spatial-merge-block order). Every number reported before
   `vision_reference.py` existed meant "the card agrees with this file", not "this file is
   right".
2. **The harness had a second, stale reference.** The device's `position_bias` built its
   position input in raster order while everything else was in block order, and the `embed`
   check passed at 2.5e-07 because its reference used the same wrong array. What caught it was
   `ln1` at 4.13e-01 while `embed` was green - a stage that is green while its consumer is red
   is a reference bug, not a device bug.
3. **The acceptance test was invalid.** Colour probes were scored at 8-10 generated tokens,
   which for a reasoning model is only the preamble. HF itself, loaded in transformers and
   given the same images, emits 'The user wants a one-word answer identifying' for a red
   square and for a blue one - identical. The probe could not distinguish anything, and the
   "every image is blue" conclusion drawn from it was measuring a preamble. With
   `enable_thinking=False` HF answers the split-colour mirror pair correctly ('Red' for
   red|blue, 'Blue' for blue|red).

4. **The hidden-state reader was transposed.** Every "our hidden state vs HF" number in this
   document was read with `d2h((H, T))` instead of `d2h((T, H)).T`. Checked against the
   checkpoint's own `embed_tokens` rows: the correct read matches to 0.000000, the transposed one
   correlates at 0.0026. It produced the `b230f3a` "first divergence is layer 1" finding, which is
   void, and it hid the splice bug for a full session - a correct read of the *embedding* would
   have shown the image columns permuted immediately. Caught by the same rule as the rest: the
   thing used to assert was itself unchecked.

5. **The splice uploaded the features transposed.** Declared `(H, T)`, device tensors
   token-major, so the host array is `(T, H)`; the caller sent `(H, count)`. Every value survived
   in the wrong place, so magnitude checks could not see it. This is the bug the whole verdict
   turned on: 2/19 to 19/19.

6. **The "2.0e-01 fidelity gap" was an instrument.** HF's end-to-end `get_image_features` runs
   bf16; HF's own vision model in f32 agrees with ours to 2.3e-04 on the same images, and HF's
   two paths differ from each other by as much as we supposedly did. The gap was recorded as a
   tower bug with a suspected wrong attention mask; reading the code cleared the mask
   (`is_causal=False`) and re-running the *same* instrument on both sides cleared the tower.

7. **Decode positions were uncompressed after an image.** Not an instrument failure but a real
   bug the acceptance test could not see, because the acceptance test scores the first token and
   the first token comes from the prefill. It took a 48-token continuation against HF to make it
   visible (13/48 -> 47/48). The general form: a check that stops at the first token cannot test
   anything that only goes wrong afterwards.

The transferable lesson, which cost more than any single bug: **three of four hypotheses were
wrong, each tested by looking for the symptom rather than measuring the thing itself. The two
checks that settled the question took one run each.** Verify the claim, not just its
components - and when a check passes, make sure the check itself was checked.

## Which tower runs

`RESIN_GAUDI_VISION=device|cpu`, read by `media.tower_from_checkpoint`. The default is
`device`, which is what the 19/19 was measured with and is 3-20x faster than the oracle
(19-74 ms against 181-1241 ms per image). `cpu` still works and is worth keeping for
comparison: it needs no card, so it can run where the device path cannot. Both towers expose
`forward(pixel_values, grid_thw)` and `config.merge`, which is all `prepare_image_prompt` uses,
so nothing above `media.py` knows which one it has. The runner hands the tower the engine's
device, so there is one handle per server rather than two.

## Verified

Both towers through `prepare_image_prompt`, same pixel values and rope config, diffing the
`ImagePrompt` the runner consumes, at the checkpoint's full depth of 27:

| grid | patches | image tokens | device | cold compile | warm load | oracle | relative |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8x8 | 64 | 16 | 11.4 ms | 1.25 s | 20.3 ms | 235 ms | 2.92e-06 |
| 16x16 | 256 | 64 | 18.8 ms | 1.27 s | 21.0 ms | 181 ms | 1.73e-06 |
| 32x32 | 1024 | 256 | 74.3 ms | 2.06 s | 22.7 ms | 1241 ms | 1.11e-05 |

"device" is h2d + launch + sync + d2h, oracle excluded. "cold compile" is a fresh
graph name; "warm load" is the same shape again, which `RecipeCache` keeps on
disk. Loading and uploading the checkpoint is 5-15 s, once.

Per-stage at depth 1 with the real weights: patch embed 2.6e-07, LayerNorm
6.5e-07, q 3.5e-07, scores 4.5e-07, the 16-head sum 4.6e-07, block 0 8.2e-07.

The engine path, with the device tower, at the checkpoint's full depth:

| check | result |
| --- | --- |
| accuracy set (`vision_acc_ours.py`) | **19 / 19** (was 2 / 19) |
| the same set through a real server, module 6, port 8011 | **19 / 19**, 1.3 s median with MTP on |
| embed's image columns vs the tower's features | **0.0000, exact** (was 85.7488, permuted) |
| a text prefill after a media prefill vs before it | **bit-identical** |
| a 2521-token text prompt (bucket 4096, chunked `kv_rows`) | greedy 8/8 match with HF |
| 48 greedy tokens after an image, vs HF | 47/48, 48/48, 47/48 |
| four non-256 images through the server (upscale, rounding, extreme aspect, downscale) | 4/4 |
| the prompt's mrope delta vs HF's `rope_deltas` | identical (-56) on every case |
| the tower vs HF's vision model in f32, real images | 2.3e-04 at the merger |
| preprocessing `pv` vs HF's processor, identity path | bit-identical |
| preprocessing `pv` vs HF's processor, four resampling branches | 1 level or better (was 18) |

`models/gaudi/tests/test_qwen_vision_rope_binding.py` runs the four media-plan assertions under pytest:
it changes nothing but the rope rows and requires the answer to change (5.88 / 1.65 / 4.68 max
logit difference for mrope, zeros and the standard table - those were 17.87 / 4.22 / 13.65 while
the splice was permuting the features, so they are a property of the prompt, not of the rope),
then asserts what the plan binds - the pre launch's `cs_tab` is the media buffer, the text buffer
still holds the standard table, the media buffer holds this prompt's rows.

## The tower's fidelity

The tower is faithful. On real images - the processor's own `pixel_values`, so there is no
preprocessing question in the way - it agrees with HF's vision model at the merger:

| image | patch embed + position | after the last block | merger output |
| --- | --- | --- | --- |
| 16x16 synthetic gradient | 6.8e-07 | 2.0e-05 | 1.5e-04 |
| solid red | 4.1e-07 | 2.2e-05 | **2.3e-04** |
| split red\|blue | 4.1e-07 | 1.2e-05 | **2.3e-04** |

### VOID: the "2.0e-01 fidelity gap"

For a while this section recorded a 2.0e-01 relative gap on one image, with the device tower
blamed and a wrong attention mask offered as the likely cause. It was the instrument. The
comparison ran HF's *end-to-end* `get_image_features` in `.venv-ref`, which dequantizes the
checkpoint to bf16 and runs the tower in bf16; measured against HF's own vision model in f32 the
same images give 2.3e-04. HF's f32 tower and its bf16 end-to-end path differ from *each other* by
8.8e-02 (solid) and 2.0e-01 (split), with the bf16 path reading larger (95.0 against 91.4, 100.5
against 85.4). So the gap was between two HF execution paths, and our tower sits on the f32 side
of it.

The mask suspect was checked anyway, by reading: `vision.py`'s attention is
`scaled_dot_product_attention(..., is_causal=False)`, which is the full bidirectional attention
the tower needs, so image patches already attend to each other in both directions.

The transferable part: this is the fourth time an input or an instrument that could not
distinguish the hypothesis was used to clear or convict something. A bf16 execution path cannot
adjudicate an f32 tower, and a uniform image cannot detect a mask error - both look like
agreement.

### Preprocessing: the resize branch, found and fixed

Preprocessing matches HF's `image_processor` on every resize branch, after a fix. The claim
that it was already bit-identical was true only of the input class that does no work.

`min_pixels` is 65536 = 256 squared, so a **256x256 image takes the identity path**: neither the
upscale nor the downscale branch fires and no filter runs. Every other size takes a branch that
does. Measured `pv` against HF's processor:

| input | branch | before | after |
| --- | --- | --- | --- |
| 512x384 | identity, with structure | bit-identical | bit-identical |
| 100x100 | below `min_pixels`: upscale | 1.4e-01 (18 levels) | **7.8e-03 (1 level)** |
| 1000x64 | `min_pixels`, extreme aspect | 3.9e-03 | **6e-08** |
| 300x500 | no branch, non-square rounding | 3.1e-03 | **6e-08** |
| 6000x6000 | above `max_pixels`: downscale | 5.6e-02 | **7.8e-03 (1 level)** |

The resized dimensions and grids matched on all five branches before the fix as well, so
`resize_shape` was never at fault - the rounding, the `floor` in the max branch, the `ceil` in
the min branch and the ties-to-even all agree with HF's `smart_resize`. The filter was.

The cause, found by matching torchvision rather than by reasoning about it: the cubic
coefficient is **not** the difference. torchvision's bicubic is a = -0.5, the same Catmull-Rom
the `image` crate's `FilterType::CatmullRom` uses, and a float reimplementation of the separable
form matches `tvF.resize` on a float tensor to 0.0000. What differs is that HF's processor
resizes **uint8** tensors, so torchvision quantizes to 8 bits *between the two separable
passes* as well as at the end. Without that intermediate quantization the cubic's overshoot
reaches the second pass, and on a 2.5x upscale of a black circle on white the two disagree by up
to 18 levels.

`resize_torchvision_bicubic` (`crates/resin-recipe/src/media/mod.rs`) is the separable form in
8-bit levels with the intermediate quantized; Qwen's `resample` default is 4 (torchvision
bicubic) instead of 3 (the `image` crate's CatmullRom). The remaining 1 level on the two
scaling branches is 8-bit quantization at the rounding boundary.

`models/gaudi/tools/vision_preprocess.py` is the probe: five branches, the resized size against `resize_shape`, and
no value outside the source range. `qwen.rs` has a Rust test for the intermediate quantization,
because that is the part that would regress silently. `models/gaudi/tests/test_qwen_vision_preprocess.py`
runs the five branches under pytest and adds the two checks the probe lacked: every value must
land on an 8-bit level, which the pre-fix float path fails on 66% of the values of the 100x100
case, and a fingerprint per branch that fails if the path changes at all.

### Two tower numbers worth keeping

Comparing the two tower *implementations* (not the model), at depth 27:

| image tokens | logits relative difference | greedy token |
| --- | --- | --- |
| 64 | 3.8e-02 | same |
| 256 | 6.8e-02 | same |

That is the card's tower against the CPU tower, whose features agree to 8.1e-06 and 4.1e-06.
The logit gap is that difference amplified through 27 FP8 layers, not a wiring fault: the same
features reproduce bit-identically, and a text prefill after the image requests is bit-identical
to a text baseline taken before them. The greedy token is the acceptance criterion; the logit
delta is a documented property.

## Two reasons a patch grid cannot be padded

The graph's tensors are static, so a recipe is compiled per `(height, width)`
patch grid, and a new image size is a new compile. The tempting optimisation is
to round the grid up to a fixed bucket set and pad. Do not, for two independent
reasons:

1. **The attention has no mask.** It is bidirectional over every patch, so a
   padded patch is a real token that every real patch attends to. The scores of
   the real patches change.
2. **The position embeddings are interpolated to the actual grid.** The tower
   bilinearly resamples the 48x48 table to `(height, width)`, so padding the grid
   changes the interpolated positions of the *real* patches as well.

Either one alone is disqualifying. The honest route to a bounded compile cost is
a masked variant of `rg_softmax_rows_f32` (or a length argument to it), after
which padding becomes safe. It is not a priority: `RecipeCache` persists recipes
per graph name, so a size costs 1.3-2.1 s once per host and ~21 ms thereafter.

## One image per prompt

`DeviceTower.forward` and `VisionTower.forward` both handle a single image and
raise on more, rather than quietly using the first grid. The recipe sums image
tokens over all images in `image_grid_thw`, so enabling multi-image means looping
the tower per image and concatenating, on both sides.

## Three things that were believed and are not true

Recorded because each one looked like evidence and cost time. If you are reading
this because something "reads back as zeros", check these first.

1. **"The first GEMM produces nothing; a GEMM feeding a pooled tensor reads back
   as zeros."** No. The harness returned from its stage branch *before* the
   weight-upload loop, so `patch_embed.proj.weight` was still the zero-filled
   allocation, and a zero-weight GEMM is exactly zero. Every downstream
   intermediate was zero for the same reason. The readback was measuring the
   harness.
2. **"Two TPC nodes in a row misbehave before a GEMM"**, and "an elementwise add
   between the GEMM and the norm leaves part of the next GEMM's output
   unwritten". Same origin. TPC chains and GEMM fan-out are fine; the tower relies
   on both.
3. **"The device disagrees with the oracle from attention onward."** It did, and
   the **oracle** was wrong: `_attention` fed `scaled_dot_product_attention` q/k/v
   shaped `(tokens, heads, head_dim)`, and SDPA takes the last two dims as
   `(sequence, head_dim)`, so it attended across *heads* instead of across
   patches. Its output then did `permute(1, 0, 2).reshape(tokens, hidden)`, which
   scrambles the head concatenation. Both are shape-preserving and produce finite,
   plausible values.

The lesson worth keeping: when the device and the oracle disagree, settle it with
an *independent* reference rather than by adjusting either side. What settled
this was computing the 16-head sum in explicit numpy — 4.6e-07 agreement with the
card — and then asking why the oracle disagreed with both.

## Findings that are real

- **Tensors are feature-major.** A tensor declared `(d0, d1)` keeps dim 0
  contiguous, so a host array binds transposed (`arr.T`) and reads back as
  `d2h(reversed(shape)).T`. This rule is right and was already written down here, and the
  image path still violated it: `splice` takes a `(H, T)` tensor, so its host array is
  `(T, H)` - the *transpose of the features*, padded to T rows - and the caller sent the
  features' own `(H, count)` shape. Read the rule as applying to every tensor including the
  ones whose shape looks like the natural host shape; that is where it gets broken.
- **Custom kernels register as `rg_<stem>_f32`**, and the file stem must equal
  the middle part: `add.c` gives `rg_add_f32`.
- **Kernel geometry comes from `geom_for()`**; an unregistered kernel falls back
  to `geom_out_tokens`, i.e. one program per column, index space `[T]`.
- **Graph identity is thorough**: `tensor()` records `T|name|dtype|shape|persistent`, node
  `params` bytes are folded in, and `identity()` adds every kernel digest plus the glue,
  runtime and Synapse build and the compiler environment. A new shape still needs a new graph
  *name* - but because the artifact filename is `{graph.name}-{identity[:32]}.recipe`, so the
  name is what separates two shapes on disk, not the hash. (An earlier note here claimed the
  hash omitted shapes; that was wrong.)
- **The attention scale `1/sqrt(head_dim)` is in no kernel.** Fold it into q on
  the host, weight and bias together, and divide rather than multiply by a
  rounded reciprocal.
- **`head_dim` 72 is not a multiple of 64**, which the 64-lane kernels want, so
  each head is zero-padded to 128. The padding is zeros, so it contributes
  nothing to the scores, the values or the projection.

## What is wired now

The preset serves images. `media=True` and `image_warmup=True` on `qwen3_5-gaudi2`, the tower on
the card by default, and a real server on module 6 answering the accuracy set 19/19 at a 1.3 s
median per request with MTP on (the per-case table is at the top of this document; the earlier
0.8 s was the same set with MTP off). The four integration points this
section used to list are all in:

1. **Splice** - `rg_splice_f32` sits in `embed_graph` between `cast_bf16_to_f32` and
   `add_rms8`, as a separate graph variant (`qwen35_embed_T{n}_img`) so a text prefill never
   carries the node. The host layout is `image_block()`, and that function is the bug that made
   image support not work.
2. **Plan selection** - `media is not None` is part of the plan key, so a text plan is never
   reused for an image prompt, and the embed step takes `extra={"img", "img_span"}`.
3. **Rope** - a media plan binds `cs_tab` to a dedicated `cs_tab_media` buffer, uploaded
   blocking, instead of overwriting the text table and restoring it. `pos = arange(T)` selects
   the row, so mrope needs no graph change. The old "restore afterwards" plan was the second
   real bug.
4. **Span and decode positions** - the span comes from the media payload's
   `image_positions` (validated contiguous in the Rust payload builder), and
   `RowCache.pos_delta` carries the prompt's mrope delta into every later step (a
   per-sequence map in the runner, cleared for a fresh prompt and on release). The span is
   not found by scanning the token ids for a run of identical placeholders: the runner only
   sees one prefill chunk's tokens, and a chunk boundary - or a prefix-cache restore - can
   start or end inside the run, which is the "0 maximal runs of 729" crash this replaced.

What is still open:

- **MTP with media is skipped, not wired.** The preset runs MTP (`draft_tokens=2`), so this
  path is reachable. An image prompt now skips `_prefill_mtp` entirely and the runner decodes
  that sequence without speculation; text sequences keep MTP. Wiring it properly means giving
  the MTP layer the prompt's mrope rows - and because its columns are shifted by one from the
  target's, a chunk after the first would need the previous chunk's last row, which the media
  table no longer holds. That is real work, and drafts are an optimisation, so the honest
  version is the skip. Note what this replaced: the first attempt raised, and a raise inside an
  engine step takes the whole server down rather than degrading to a wrong answer - the guard
  has to be a routing decision, not an exception.
- **One image per prompt.** Unchanged from before; see "One image per prompt" below.

## Running it

The harnesses and the checkpoint-gated tests read `RESIN_GAUDI_QWEN35`: the
directory of a Qwen3.8-27B FP8 checkpoint (see `models/gaudi/README.md`).

```
# the oracle against HF's tower, stage by stage (needs transformers)
RESIN_GAUDI_QWEN35=/path/to/Qwen3.8-27B-FP8 \
  PATH=/opt/habanalabs/bin:$PATH ./.venv/bin/python models/gaudi/tools/vision_reference.py

# the preprocessing against HF's image processor, one case per resize branch
RESIN_GAUDI_QWEN35=/path/to/Qwen3.8-27B-FP8 \
  HABANA_VISIBLE_MODULES=6 PATH=/opt/habanalabs/bin:$PATH \
  .venv/bin/python models/gaudi/tools/vision_preprocess.py

# the 19-case accuracy set: through the engine, then through HF in its own interpreter
RESIN_GAUDI_QWEN35=/path/to/Qwen3.8-27B-FP8 \
  HABANA_VISIBLE_MODULES=6 .venv/bin/python models/gaudi/tools/vision_acc_ours.py
RESIN_GAUDI_QWEN35=/path/to/Qwen3.8-27B-FP8 \
  .venv-ref/bin/python models/gaudi/tools/vision_acc_hf.py
```
