#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
ulimit -s 67108864
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --trust-remote-code --host 0.0.0.0 --port "$PREFILL_PORT" \
  --random-seed 42 --context-length 1048576 --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --warmups prefill_input_shapes --disable-shared-experts-fusion --disable-piecewise-cuda-graph \
  --disable-chunked-prefix-cache --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --mem-fraction-static 0.8 --max-running-requests 16 --max-mamba-cache-size 96 \
  --mamba-scheduler-strategy extra_buffer --page-size 64 --disable-cuda-graph --enable-single-batch-overlap \
  --tp-size 8 --ep-size 8 --attn-cp-size 8 --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split --moe-a2a-backend deepep --deepep-mode normal \
  --attention-backend nsa --nsa-prefill-backend flashmla_auto --nsa-decode-backend flashmla_kv \
  --kv-cache-dtype fp8_e4m3 --numa-node 0 3 2 1 4 7 6 5 \
  --enable-hierarchical-cache --hicache-size 220 --hicache-write-policy write_through \
  --hicache-io-backend kernel --hicache-mem-layout layer_first --speculative-algorithm EAGLE \
  --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --enable-cache-report --enable-metrics --tokenizer-worker-num=8 \
  --deepep-config "$(dirname "$0")/ep_config.json" \
  --json-model-override-args '{"index_share_for_mtp_iteration": true}'
