//! Conversion only: request planning and validation live in resin-core.
use crate::{BatchPhase, BatchPlan, RequestPayload, cache::CacheGroupPlan};
use resin_core::{Phase, engine::ExecutionPlan};
use std::sync::OnceLock;

pub(super) fn from_core(plan: ExecutionPlan<RequestPayload>) -> BatchPlan {
    // A single-row prefill carries its full prompt so the runner can prefetch the next chunk's Engram
    // gather across forwards; decode (and multi-row plans) carry none, leaving the decode path untouched.
    let prefill_prompt = if matches!(plan.phase, Phase::Prefill) && plan.payloads.len() == 1 {
        plan.payloads[0].prompt.as_ref().clone()
    } else {
        Vec::new()
    };
    let mut sampling_params = Vec::with_capacity(plan.payloads.len());
    let mut media = Vec::with_capacity(plan.payloads.len());
    for payload in plan.payloads {
        sampling_params.push(payload.sampling_params);
        media.push(payload.media);
    }
    // The core constructs this plan from validated request and allocator state.
    // Public Python constructors still validate untrusted caller-provided arrays.
    BatchPlan {
        prompt_score_indices: plan.prompt_score_indices,
        prompt_score_token_ids: plan.prompt_score_token_ids,
        prompt_score_indptr: plan.prompt_score_indptr,
        prompt_score_top_counts: plan.prompt_score_top_counts,
        prompt_target_indptr: plan.prompt_target_indptr,
        prompt_target_token_ids: plan.prompt_target_token_ids,
        sequence_ids: plan.sequence_ids,
        context_lengths: plan.context_lengths,
        prefill_lengths: plan.prefill_lengths,
        token_ids: plan.token_ids,
        token_indptr: plan.token_indptr,
        prefill_prompt,
        phase: match plan.phase {
            Phase::Prefill => BatchPhase::Prefill,
            Phase::Decode => BatchPhase::Decode,
        },
        batch_id: plan.batch_id,
        batch_key: plan.batch_key,
        cache_groups: plan
            .cache_groups
            .into_iter()
            .map(CacheGroupPlan::from_inner)
            .collect(),
        sampling_mask: plan.sampling_mask,
        sampling_params,
        media,
        positions: OnceLock::new(),
    }
}
