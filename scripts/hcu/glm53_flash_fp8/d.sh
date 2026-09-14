#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
ulimit -s 67108864
export SGLANG_NSA_FUSE_TOPK=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=96
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --trust-remote-code --host 0.0.0.0 --port "$DECODE_PORT" \
  --dist-init-addr "$DECODE_HOST:2000" --max-running-requests 64 --random-seed 42 \
  --disable-shared-experts-fusion --disable-piecewise-cuda-graph --disable-chunked-prefix-cache \
  --disable-radix-cache --cuda-graph-bs 1 2 4 6 8 10 12 --mem-fraction-static 0.85 \
  --context-length 1048576 --page-size 64 --dtype bfloat16 --max-mamba-cache-size 320 \
  --mamba-scheduler-strategy no_buffer --tp-size 8 --dp-size 8 --ep-size 8 --moe-dense-tp-size 1 \
  --enable-dp-attention --enable-dp-lm-head --moe-a2a-backend deepep --deepep-mode low_latency \
  --attention-backend nsa --nsa-prefill-backend flashmla_auto --nsa-decode-backend flashmla_kv \
  --linear-attn-backend triton --kv-cache-dtype fp8_e4m3 --quantization w8a8_fp8 \
  --disaggregation-transfer-backend mooncake --disaggregation-mode decode \
  --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 --reasoning-parser glm5 --tool-call-parser glm5stream \
  --enable-cache-report --enable-metrics --tokenizer-worker-num=8 \
  --json-model-override-args '{"index_share_for_mtp_iteration": true}' --numa-node 0 3 2 1 4 7 6 5 \
  --deepep-config "$(dirname "$0")/ep_config.json"
