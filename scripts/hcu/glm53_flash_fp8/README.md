# GLM-5.3-Flash FP8 on HCU

These recipes target the `/home/work/GLM-5.3-Flash-Channel-FP8-w8a8` checkpoint
used by the `dev-wl-glm2` container. Copy this directory to `/home/work/glm`
after updating `/home/work/code/sglang-das` from the commit containing it.

* `ifb.sh` starts a single-node baseline server on eight cards. It uses
  DeepEP `auto` mode so extend/prefill batches take the normal contiguous
  path while decode batches take the low-latency masked path, matching the
  split used by `p.sh` and `d.sh`.
* `p.sh` starts the eight-card prefill role.
* `d.sh` starts the eight-card decode role (`TP8/DP8/EP8`); this is deliberately
  the temporary one-node decode layout requested for bring-up.
* `router.sh` fronts the two PD roles on port 30005.

Override `MODEL_PATH`, `PREFILL_HOST`, `PREFILL_PORT`, `DECODE_HOST`,
`DECODE_PORT`, or the HCA variables before launching if the deployment uses
different addresses. The recipes intentionally keep the remote reference's
HCA and topology defaults; no NIC-specific values are embedded in launch
arguments.

Example baseline check:

```bash
cd /home/work/glm
bash ifb.sh
```

Then query `http://<host>:8080/v1/chat/completions` with the model name
`default`. For PD, run `bash p.sh`, `bash d.sh`, and `bash router.sh` in
separate shells and query the router at port 30005.
