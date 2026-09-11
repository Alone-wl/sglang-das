"""GLM speculative streaming using main's request-aligned IPC records."""

import functools

import msgspec

from sglang.srt.managers.io_struct import unwrap_from_pickle, wrap_as_pickle
from sglang.srt.runtime_context import get_serving


def _slice_token(value, step):
    if value is None or isinstance(value, (str, bytes, dict)):
        return value
    try:
        return value[step : step + 1]
    except (TypeError, IndexError):
        return value


def interleave_batch_token_id_out(func):
    @functools.wraps(func)
    def wrapper(self, recv_obj):
        if not get_serving().glm_stream_speculated_tokens or not recv_obj.rids:
            return func(self, recv_obj)

        # Preserve whole records for newly observed, completed, beam, or
        # versioned requests. Only established decode streams are split.
        split = [
            rid in self.decode_status
            and recv_obj.finished_reasons[i] is None
            and not (recv_obj.beam_search_output and recv_obj.beam_search_output[i])
            and not (recv_obj.weight_versions and recv_obj.weight_versions[i])
            for i, rid in enumerate(recv_obj.rids)
        ]
        lengths = [
            len(ids) if split[i] else 1 for i, ids in enumerate(recv_obj.decode_ids)
        ]
        max_steps = max(lengths, default=1)
        if max_steps <= 1:
            return func(self, recv_obj)

        outputs = []
        token_fields = {
            "decode_ids",
            "output_ids",
            "output_token_logprobs_val",
            "output_token_logprobs_idx",
            "output_top_logprobs_val",
            "output_top_logprobs_idx",
            "output_token_ids_logprobs_val",
            "output_token_ids_logprobs_idx",
            "output_token_entropy_val",
            "output_hidden_states",
            "routed_experts",
            "indexer_topk",
            "token_steps",
            "output_token_sampling_mask",
            "output_token_sampling_logprobs",
        }
        wrapped_fields = {"time_stats", "customized_info"}
        for step in range(max_steps):
            active = [i for i, length in enumerate(lengths) if step < length]
            if not active:
                continue
            updates = {}
            for name in recv_obj.__struct_fields__:
                value = getattr(recv_obj, name)
                if name in wrapped_fields:
                    value = unwrap_from_pickle(value)
                if value is None:
                    continue
                if name == "customized_info":
                    selected = {
                        key: [
                            _slice_token(items[i], step) if split[i] else items[i]
                            for i in active
                        ]
                        for key, items in value.items()
                    }
                elif isinstance(value, list):
                    # Optional counters can be empty even in a nonempty batch.
                    selected = [value[i] for i in active if i < len(value)]
                    if name in token_fields:
                        selected = [
                            _slice_token(value[i], step) if split[i] else value[i]
                            for i in active
                            if i < len(value)
                        ]
                    elif name.startswith("input_") and step:
                        selected = [
                            None if name.endswith("_flat") else [] for _ in selected
                        ]
                else:
                    continue
                updates[name] = (
                    wrap_as_pickle(selected) if name in wrapped_fields else selected
                )

            updates["decoded_texts"] = ["" for _ in active]
            updates["completion_tokens"] = [
                recv_obj.completion_tokens[i] - (lengths[i] - 1 - step) for i in active
            ]
            if recv_obj.spec_verify_ct:
                updates["spec_verify_ct"] = [
                    min(recv_obj.spec_verify_ct[i], updates["completion_tokens"][j])
                    for j, i in enumerate(active)
                    if i < len(recv_obj.spec_verify_ct)
                ]
            outputs.append(func(self, msgspec.structs.replace(recv_obj, **updates)))
        return outputs

    return wrapper
