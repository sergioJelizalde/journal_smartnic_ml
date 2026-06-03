# SmartNIC ML — ONNX / DPDK Inference Pipeline

End-to-end pipeline for training a flow-level DDoS classifier and deploying it
on a BlueField-3 SmartNIC using ONNX Runtime inside a DPDK multicore application.

```
PCAPs  ──►  feature_extraction.ipynb  ──►  Benign_flows.csv / DDoS_flows.csv
                                                        │
                                         mlp_pytorch_onnx_pipeline.ipynb
                                          ├── PyTorch training (GPU/CPU)
                                          ├── C headers  →  scalar / ARM NEON inference
                                          ├── ONNX export  →  mlp_onnx_models/*.onnx
                                          └── mlp_ort_dpdk.h  →  DPDK lcore integration
```

---

## Notebooks

### `feature_extraction.ipynb`

Converts raw PCAP files into per-flow feature CSVs using **tshark** and **pandas**.

**Inputs**
| Path | Contents |
|---|---|
| `D:\datasets\pcaps_cic\*.pcap/pcapng` | Benign traffic captures |
| `D:\datasets\pcaps_cic\ddos_*.pcap` | DDoS traffic captures |
| `D:\datasets\pcaps_cic\dos\DoS_*.pcap` | DoS / Recon captures |

**Processing steps**
1. tshark extracts per-packet fields: timestamp, src/dst IP, ports, TCP flags, `ip.len`, `frame.len`
2. Packets are grouped into 5-tuple flows `(src_ip, dst_ip, src_port, dst_port, proto)`
3. For each flow, sliding windows of N = {4, 8, 16, 32, 64, 128} packets are computed
4. Per-window features extracted:

| Feature | Description |
|---|---|
| `iat_mean / iat_min / iat_max` | Inter-arrival time statistics |
| `ip_len_mean / ip_len_min / ip_len_max` | IP payload length statistics |
| `frame_len_mean / frame_len_min / frame_len_max` | Frame length statistics |
| `tcp_flag_count` | Count of packets with TCP flags set |
| `bytes_sent` | Sum of `ip_len` over the window |

**Outputs**
```
D:\datasets\pcaps_cic\flow_features\
  ├── Benign_flows.csv   # labelled 0
  └── DDoS_flows.csv     # labelled 1
```

> CSVs are excluded from git via `.gitignore` (files exceed GitHub's 100 MB limit).
> Generate them locally by running all cells in order.

**Dependencies**
```
pip install pandas
# tshark must be installed: https://www.wireshark.org/download.html
# tshark path: C:\Program Files\Wireshark\tshark.exe
```

---

### `mlp_pytorch_onnx_pipeline.ipynb`

Trains five MLP architectures on the 8-feature flow dataset, exports weights for
C-based inference (scalar / ARM NEON), then exports to ONNX and generates a
ready-to-use DPDK multicore integration header.

**Inputs**
```
D:\datasets\pcaps_cic\flow_features\
  ├── Benign_flows.csv
  └── DDoS_flows.csv
```
Only rows with `N == 8` are used (8-packet windows → 8 input features).

**Features used for training**

```
iat_mean, iat_min, iat_max,
ip_len_mean, ip_len_min, ip_len_max,
tcp_flag_count, bytes_sent
```

**MLP architectures trained**

| Name | Hidden layers | Params |
|---|---|---|
| `mlp_8` | 8→8→2 | ~100 |
| `mlp_32` | 8→32→2 | ~320 |
| `mlp_64_32` | 8→64→32→2 | ~2 k |
| `mlp_128_64_32` | 8→128→64→32→2 | ~7 k |
| `mlp_256_128_64_32` | 8→256→128→64→32→2 | ~28 k |

Training: Adam, CrossEntropyLoss, 30 epochs, batch 256. GPU used when available.

**Outputs**

| File | Description |
|---|---|
| `mlp_headers_demo/mlp_*.h` | C headers with weights/biases for scalar/NEON inference |
| `feature_stats.h` | z-score mean/std for runtime normalisation |
| `mlp_onnx_models/mlp_*.onnx` | ONNX models (opset 17, dynamic batch axis) |
| `mlp_ort_dpdk.h` | DPDK multicore ORT integration header (see below) |

**ONNX output tensor**

```
input  : "float_input"   float32  [batch, 8]   z-score normalised
output : "logits"        float32  [batch, 2]   argmax → 0 Benign / 1 DDoS
```

**Dependencies**
```
pip install torch torchvision scikit-learn pandas onnxruntime
```

---

## DPDK Integration — `mlp_ort_dpdk.h`

Drop-in header for a DPDK lcore worker. Each lcore owns its own `OrtSession`
(no locking) with `IntraOpNumThreads=1` and full graph optimisation enabled.
Logits land in a **static per-lcore buffer** — zero heap allocation in the
RX hot path.

```c
// lcore entry point
static int lcore_worker(void *arg)
{
    mlp_ort_lcore_init();               // create OrtSession for this lcore

    while (running) {
        n = rte_eth_rx_burst(port, q, pkts, MLP_MAX_BATCH);
        extract_features(pkts, feats, n);          // fill float[]
        normalize(feats, n, FEATURE_MEAN,          // feature_stats.h
                            FEATURE_STD);
        mlp_ort_infer(feats, preds, n);            // 0=Benign  1=DDoS
        dispatch(pkts, preds, n);
    }

    mlp_ort_lcore_cleanup();
    return 0;
}
```

**Build flags**
```makefile
CFLAGS += -I$(ORT_ROOT)/include
LDFLAGS += -L$(ORT_ROOT)/lib -lonnxruntime
```

**ARM NEON comparison**
The headers in `mlp_headers_demo/` provide the same model weights for a hand-written
scalar / NEON C inference path. Run both paths on the BF3 to compare throughput
against ORT.

---

## Files tracked in git

```
onnx/
├── README.md
├── feature_extraction.ipynb
├── mlp_pytorch_onnx_pipeline.ipynb
├── Makefile
└── iot_multicore.c
```

> `*.csv`, `*.onnx`, `*.pt`, `*.pth` are in `.gitignore`.
> Generated headers (`mlp_headers_demo/`, `feature_stats.h`, `mlp_ort_dpdk.h`)
> are regenerated by running the notebook.