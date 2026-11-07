# Multiple images per conversation

**Status: planned, not implemented.** One image per prompt works, including across
prefill chunks and prefix-cache restores (`docs/vision.md`). This document is the
plan for lifting the single-image limit, so that a conversation can carry several
images: an early image, text back-and-forth, the same image again, a different
image. Today any render with more than one image instance is refused at render
time with a 400 ("this preset serves one still image per prompt") - deliberately,
because the alternative was an engine-step exception that kills the server.

## Why it is mostly plumbing

The upstream machinery already handles N images:

- The Rust payload builder (`crates/resin-py/src/media_payload.rs`,
  `validate_packed_images`) validates N ordered, grid-consistent placeholder
  ranges and computes mrope positions across all of them
  (`crates/resin-recipe/src/media/qwen.rs`, `mrope_positions`).
- The media payload carries every placeholder position in `image_positions`.
- Chunking and prefix-cache restores are position-based: a chunk that holds the
  middle of an image's placeholder run is sliced correctly once the span is known
  in prompt coordinates (proven by the chunked-prefill regression test).

The single-image assumptions are all on the Wright side: the towers run one image
per `forward`, the splice kernel takes one `(first, count)` span, `slice_media`
intersects one span with the chunk, and `StillImageRecipe` rejects N > 1 at render.

## The plan

1. **Kernel: multi-span `splice.c`.** The span tensor becomes k triples
   `(first, count, img_offset)`; `img_offset` is where that span's rows start in
   the stacked `img` input. Per output column, a small loop scans the spans
   (k is at most 8, negligible); zero-count spans are skipped. `img` stays
   `(H, T)` with feature runs stacked in span order. The span input is a fixed
   `(3 * IMG_SPAN_MAX,)` I32 tensor with `IMG_SPAN_MAX = 8`, because graph input
   shapes are static; the host zero-pads the triples.
2. **Graph: `layers.py`.** `embed_graph(media=True)` binds `img_span` as
   `(3 * IMG_SPAN_MAX,)` I32 instead of `(2,)`; everything else unchanged.
3. **Towers: `vision.py`, `vision_device.py`.** `forward` splits `pixel_values`
   at the grid boundaries (cumsum of `t*h*w`), runs each image - `DeviceTower`
   reuses its per-shape recipe/scratch cache, so repeated sizes are cheap - and
   concatenates features in prompt order, which the payload guarantees is span
   order. `VisionTower.stages()` (the dev-harness taps) keeps a single-image
   assert; `forward` loops on both towers.
4. **`media.py`.** `prepare_image_prompt` splits `image_positions` into per-image
   spans at contiguity breaks (Rust already validated per-image contiguity and
   ordering) and re-checks each span's count against its grid. `ImagePrompt`
   gains `spans: tuple[(first, count), ...]`, replacing `first`.
5. **`model.py`.** `slice_media` intersects each span with the chunk and returns
   stacked chunk features plus local `(first, count, offset)` triples, or None
   when the chunk holds no image token. `image_block` places multiple runs into
   the `(T, H)` token-major block. `_splice_media` uploads the zero-padded
   triples. The media tuple becomes `(features, spans, rope, delta)`.
6. **`runner.py`.** `_vision` returns `prompt.spans`; the digest cache and the
   MTP skip (image sequences decode without speculation) are unchanged.
7. **`deploy/models.py`.** `StillImageRecipe` narrows: reject video
   (`temporal != 1`) and more than `IMG_SPAN_MAX` images - render-time 400s,
   because `prepare_image_prompt` runs in the runner, where an error has no
   per-request channel. Multiple still images become allowed.
8. **Tests.** Unit: span splitting, multi-run `image_block`, the guard. Card-gated:
   a chunked two-image conversation (history plus a new image - the served
   scenario), and a conversation test (image, text, the same image again, text, a
   different image) asserting every turn answers consistently. Live: the same
   script against a booted server, including two images in one message.
9. Update this document and the "One image per prompt" section of
   `docs/vision.md` when it lands.

## Decisions

- **Cap of 8 images per prompt**, from the static graph shape. Generous in
  practice; the 400 message names the cap.
- **Video stays rejected** at render: nothing on the Wright side handles
  `temporal > 1`.
- **`stages()` stays single-image**: it is a development harness, not a serving
  path.
- After this lands, the image-token limit is the only remaining image-side
  restriction; MTP for image sequences remains skipped by design (see
  `docs/vision.md`).
