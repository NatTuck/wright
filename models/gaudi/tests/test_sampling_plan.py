"""``sampling.plan_params`` follows ``resin.backend.sampling.batch_sampling_inputs`` on the plans
an Engine actually emits (no device needed)."""

from __future__ import annotations

import numpy as np

from resin import Engine, ForwardOutput, SamplingParams

VOCAB = 1000


class CheckingRunner:
    """Compares the Gaudi sampler inputs with Resin's for every plan; emits token 7."""

    def __init__(self) -> None:
        self.plans = 0

    def forward(self, plan) -> ForwardOutput:
        from resin.backend.sampling import batch_sampling_inputs
        from resin_models_gaudi.qwen3_5.sampling import philox_uniform, plan_params

        expected = batch_sampling_inputs(plan, VOCAB)
        got = plan_params(plan, VOCAB)
        np.testing.assert_allclose(got[:, 0], expected["temperatures"], rtol=1e-6)
        np.testing.assert_allclose(got[:, 1], expected["top_ps"], rtol=1e-6)
        top_ks = np.asarray(expected["top_ks"])
        np.testing.assert_array_equal(got[:, 2], np.where(top_ks >= VOCAB, 0, top_ks))
        uniform = philox_uniform(expected["seeds"], expected["offsets"]).astype(
            np.float32
        )
        # sampled rows carry their uniform; greedy rows (lane 4 = 0) never read theirs
        sampled = got[:, 4] > 0.5
        np.testing.assert_array_equal(got[sampled, 3], uniform[sampled])
        assert (got[~sampled, 3] == 0).all()
        sample = np.asarray(expected["sampling_mask"], bool)
        temps = np.asarray(expected["temperatures"])
        np.testing.assert_array_equal(got[:, 4] > 0.5, sample & (temps > 0))
        self.plans += 1
        rows = plan.num_sequences
        return ForwardOutput(
            token_ids=np.full(rows, 7, np.uint32),
            token_indptr=np.arange(rows + 1, dtype=np.uint32),
        )


def test_plan_params_follow_resin():
    runner = CheckingRunner()
    engine = Engine(runner, context_length=64, enable_overlap=False)
    params = [
        SamplingParams(max_output_tokens=3, temperature=0.8, top_k=7, seed=5),
        SamplingParams(max_output_tokens=2, temperature=0.0),
        SamplingParams(max_output_tokens=4, top_p=0.9, seed=(1 << 62) + 3),
        SamplingParams(max_output_tokens=2, temperature=1.3, top_k=5000),
    ]
    leases = [
        engine.submit(list(range(1, 4 + i)), sampling_params=p)
        for i, p in enumerate(params)
    ]
    while engine.has_work():
        engine.step()
    assert [len(lease.take_result()) for lease in leases] == [3, 2, 4, 2]
    assert runner.plans >= 4
