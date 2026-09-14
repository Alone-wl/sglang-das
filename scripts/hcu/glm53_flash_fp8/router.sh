#!/usr/bin/env bash
set -eo pipefail
exec python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill "http://${PREFILL_HOST:-10.6.14.14}:${PREFILL_PORT:-8080}" \
  --decode "http://${DECODE_HOST:-10.6.14.15}:${DECODE_PORT:-8080}" \
  --policy cache_aware --port "${ROUTER_PORT:-30005}"
