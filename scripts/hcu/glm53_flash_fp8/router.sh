#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "$0")/common.sh"
exec python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill "http://${PREFILL_HOST}:${PREFILL_PORT}" \
  --decode "http://${DECODE_HOST}:${DECODE_PORT}" \
  --policy cache_aware --port "${ROUTER_PORT}"
