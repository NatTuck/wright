"""A process can close its card and open it again."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not Path("/usr/include/habanalabs/synapse_api.h").exists()
    or shutil.which("tpc-clang") is None,
    reason="requires the Synapse SDK and tpc-clang",
)


def test_reopen_after_close():
    from resin_models_gaudi.synapse.runtime import SynapseError, open_device

    card = os.environ.get("RESIN_GAUDI_CARD")
    data = np.arange(1 << 18, dtype=np.float32)
    for _ in range(2):
        try:
            device = open_device(None if card is None else int(card))
        except SynapseError as error:
            pytest.skip(f"no Gaudi card available: {error}")
        address = device.malloc(data.nbytes)
        device.h2d(address, data)
        device.h2d_nb(address, data[:16])
        device.record(0)
        device.wait(0)
        np.testing.assert_array_equal(device.d2h(address, data.shape, np.float32), data)
        device.close()
