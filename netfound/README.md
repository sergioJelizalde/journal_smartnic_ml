# netFound inline tokenizer for DPDK

Online re-implementation of `pre_process_src/3_field_extraction.cpp` +
`Tokenize.py`, producing the same tensors `netFoundTokenizer.tokenize()`
would, one packet at a time, so you can feed a live flow window straight
into ONNX Runtime on your BF3 ARM cores.

## Files
- `netfound_tokenizer.h` / `.c` — the state machine + tensor export
- targets **no-payload** configs (small/base/large all use
  `netFoundNoPayloadConfig`) and **no TCP options**
  (`DefaultConfigNoTCPOptions.json` path)

## Wiring it into your existing pipeline

You already have a flow table (`rte_hash`) and direction logic from the
DGA/DoH work — reuse it. Per flow, add one `nf_flow_state_t` to whatever
struct you key by 5-tuple, and call:

```c
// on new flow (SYN, or first UDP/ICMP packet):
nf_flow_state_init(&flow->nf_state);

// on every packet belonging to this flow, in your RX loop:
uint64_t ts_ns = /* rte_rdtsc-derived wall time, or hardware timestamp */;
bool is_fwd = /* your existing direction check, e.g. src_ip == flow->client_ip */;
bool closed_a_burst = nf_process_packet(&flow->nf_state, m, ts_ns, is_fwd);

// on FIN/RST, or an inactivity timeout you already run for flow eviction:
nf_flush_current_burst(&flow->nf_state);

if (flow->nf_state.ready_for_inference /* or you decide to flush early */) {
    nf_model_input_t input;
    nf_export_model_input(&flow->nf_state, &input);
    // hand `input` to your ORT session — see below
}
```

`nf_process_packet` returns `true` every time a burst closes, if you'd
rather run inference on a sliding window (e.g. as soon as burst 6 closes)
instead of waiting for all 12 — worth considering for VPN/DoH classification
latency, since waiting for a full 12-burst window on a long-lived flow adds
real delay before your first classification.

## ONNX Runtime feed

`nf_model_input_t` gives you flat `int64_t` arrays already shaped for a
batch-size-1 ORT call:

- `input_ids`, `attention_mask`: shape `[1, 12, 73]` (flatten as
  `12*73 = 876`)
- `direction`, `bytes`, `iats`, `pkt_count`: shape `[1, 12]`
- `total_bursts`, `protocol`, `flow_duration`: scalars

Match these against the exact input names/shapes your exported
`netfound_small.onnx` graph expects — the HF model's `forward()` signature
(check `src/netFoundFinetuning.py`) determines which of these are actually
consumed vs. ignored by the classification head; don't assume all of them
are wired into the graph until you check.

## Before you trust this against the real model

This was written by reading the reference C++/Python source, not by
running it side-by-side against the reference pipeline. Before relying on
classification output:

1. **Validate byte-for-byte** against the offline pipeline: run the same
   PCAP through both `scripts/preprocess_data.py` (offline) and this
   tokenizer (fed via a PCAP replay), and diff the resulting `input_ids`/
   `bytes`/`iats`/`direction` tensors per flow. This is the single most
   important step — silent token-layout mismatches won't crash anything,
   they'll just quietly degrade or randomize classification accuracy.
2. **IPv6 is not implemented** in `nf_process_packet` — only an IPv4
   branch is here. Add an `rte_ipv6_hdr` path mirroring
   `3_field_extraction.cpp`'s v6 handling if your traffic mix needs it.
3. **TCP seq/ack anchor logic (assumption #4 in the header)** assumes you
   see the SYN. If your DPDK capture point can miss the start of a flow
   (e.g. BF3 attached mid-path after link flap, or a flow table eviction
   mid-flow), the anchor will be wrong for that flow — same edge case the
   original C++ handles via the `tcp_absolute_ack != 0` branch, but worth
   testing deliberately since it's the trickiest part of the whole thing.
4. **Flow timeout / eviction policy** isn't included — you'll want the
   same inactivity-timeout logic you already run for your DoH/DGA flow
   table to call `nf_flush_current_burst()` before eviction, so
   short-lived flows (many VPN handshake probes, in particular) still
   produce a usable — if short — window instead of silently vanishing.
