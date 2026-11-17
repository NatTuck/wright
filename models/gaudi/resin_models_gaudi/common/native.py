"""Pieces of the native executors' startup builders (``qwen3_5.native``, ``glm5_3_flash.native``) that every model
with the DFlash drafter shares: a launch's bindings with some of them moved to a step's tables or to registers,
and the drafter's block step as one native program over a block image (``resin_gaudi::qwen_draft::BlockLayout``).
"""

from __future__ import annotations

import numpy as np


def bind(program, recipe, slots, addresses, tables: dict, dynamic: dict) -> None:
    """A launch of ``recipe``: its bindings (slots, addresses), with ``tables`` (name -> address) and ``dynamic``
    (name -> operand) for the inputs it has."""
    pairs = dict(zip(np.asarray(slots).tolist(), np.asarray(addresses).tolist(), strict=True))
    moving = []
    for name, address in tables.items():
        if name in recipe.persistent:
            pairs[recipe.slot(name)] = int(address)
    for name, operand in dynamic.items():
        if name in recipe.persistent:
            pairs.pop(recipe.slot(name), None)
            moving.append((recipe.slot(name), operand))
    program.launch(recipe.handle, list(pairs), list(pairs.values()), dynamic=moving)


def block_tables(blocks: dict, bucket: int) -> dict:
    """The device addresses of a block step's tables of ``bucket`` rows in the block image."""
    layout = blocks["layouts"][bucket]
    base = blocks["image"]
    return {
        "tok": base + layout["tok"],
        "pos": base + layout["pos"],
        "anchors": base + layout["anchors"],
        "tables": base + layout["tables"],
    }


def block_program(Program, E, drafter, cfg, blocks: dict, bucket: int, width: int):
    """The drafter's block step of ``bucket`` rows for a verify of ``width`` columns as one native program (``E``
    the executor class with its registers ``R_BIMAGE``, ``R_BBYTES``, ``R_BLAUNCHES``): ``upload`` the block image's
    host copy ``R_BIMAGE`` (``R_BBYTES`` of it) then that copy's event, ``run`` the embedding, each layer's pre,
    ``R_BLAUNCHES`` slot launches and post, and the head."""
    layout = blocks["layouts"][bucket]
    base = blocks["image"]
    program = Program(E.REGISTERS)
    program.section("upload")
    program.upload(base, (blocks["host"].ctypes.data, E.R_BIMAGE, blocks["stride"]), (0, E.R_BBYTES, 1))
    program.select([(E.R_BIMAGE, 2)], [0, 1])
    for k in range(2):
        program.arm()
        program.record(blocks["events"] + k)
    program.end()
    program.section("run")
    step = block_tables(blocks, bucket)
    names = {"tok": step["tok"], "pos": step["pos"], "anchors": step["anchors"]}
    recipe, slots, addresses = drafter._step(("dfl_embed", bucket, width))
    bind(program, recipe, slots, addresses, names, {})
    for layer in range(cfg.layers):
        rows = {"kv_rows": base + layout["block_rows"] + layer * layout["block_stride"]}
        recipe, slots, addresses = drafter._step(("dfl_pre", bucket, width), layer)
        bind(program, recipe, slots, addresses, {**names, **rows}, {})
        recipe, slots, addresses = drafter._step(("dfl_slots", bucket, width), layer)
        program.take(E.R_BLAUNCHES)
        for i in range(layout["max_launches"]):
            at = base + layout["tables"] + (i * cfg.layers + layer) * layout["stride"]
            tables = {"slots": at, "k_pages": at + layout["k_pages"], "v_pages": at + layout["v_pages"]}
            bind(program, recipe, slots, addresses, {**names, **tables}, {})
        program.end()
        recipe, slots, addresses = drafter._step(("dfl_post", bucket, width), layer)
        bind(program, recipe, slots, addresses, names, {})
    recipe, slots, addresses = drafter._step(("dfl_head", bucket, width))
    bind(program, recipe, slots, addresses, names, {})
    program.finish()
    return program
