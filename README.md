# Protocol-aware SmartNIC anomaly detection framework

This folder is a refactored DPDK framework for selecting one traffic-analysis profile at runtime and one model microkernel at runtime:

- `--profile iot`: generic TCP flow features for IoT/botnet attacks.
- `--profile doh`: bidirectional TLS Application Data features for DoH, encrypted DGA, and DNS tunneling.
- `--profile ics`: Modbus/TCP request-response parsing and sensor-window features for ICS fault/anomaly detection.
- `--model mlp`: dense MLP inference, scalar or ARM NEON.
- `--model rf`: Random Forest inference from a static C header.
- `--model bnn`: binarized neural network inference using XNOR-popcount microkernels.

The hot path is worker-local: RX burst, parse, profile feature update, feature-contract check, normalization, model inference, action, TX burst. Logging is optional. In `make perf`, logging, timing, and feature-contract checks are compiled out.

## Build

On a BlueField/ARM DPDK system:

```bash
export PKG_CONFIG_PATH=/path/to/dpdk/lib/pkgconfig:$PKG_CONFIG_PATH
make clean
make perf
```

Instrumentation/log build:

```bash
make log
```

## Run examples

Use DPDK EAL arguments before `--`, and application arguments after `--`.

IoT TCP profile with NEON MLP, no logs:

```bash
sudo ./smartnic_ad -l 0-7 -n 4 -- --profile iot --model mlp --kernel neon --window 8 --port 0 --swap-mac 1 --allow-feature-pad 1
```

DoH/DGA/tunneling profile with async logging on a reserved lcore:

```bash
sudo ./smartnic_ad -l 0-7 -n 4 -- --profile doh --model mlp --kernel neon --window 16 --log-mode async --log-file doh_predictions.csv --port 0 --swap-mac 1
```

ICS/Modbus sensor profile with Random Forest:

```bash
sudo ./smartnic_ad -l 0-7 -n 4 -- --profile ics --model rf --ics-window 32 --port 0 --swap-mac 1
```

BNN profile:

```bash
sudo ./smartnic_ad -l 0-7 -n 4 -- --profile doh --model bnn --window 16 --port 0 --swap-mac 1
```

## Model export

The placeholder headers in `models/` compile but are not trained models. Replace them with generated headers.

MLP:

```bash
python3 tools/export_mlp.py --csv features.csv --label label --features f0,f1,f2,f3,f4,f5,f6,f7 --hidden 64,32 --epochs 30 --out models/model_mlp.h
```

Random Forest:

```bash
python3 tools/export_rf.py --csv features.csv --label label --trees 25 --max-depth 8 --out models/model_rf.h
```

BNN:

```bash
python3 tools/export_bnn.py --csv features.csv --label label --hidden 64,32 --epochs 30 --out models/model_bnn.h
```

For correctness, the generated model input dimension must match the selected profile. The framework checks this at startup. `--allow-feature-pad 1` is only for quick smoke testing.

## Feature schemas

### IoT TCP profile, 8 features

1. `pkt_size_min`
2. `pkt_size_max`
3. `pkt_size_mean`
4. `iat_min_us`
5. `iat_max_us`
6. `iat_mean_us`
7. `total_bytes`
8. `tcp_flag_bits_sum`

### DoH bidirectional profile, 16 features

1. `client_pkt_max`
2. `n_client`
3. `bytes_fraction_client`
4. `n_server`
5. `pkt_fraction_client`
6. `client_bytes`
7. `server_pkt_max`
8. `size_min`
9. `size_mean`
10. `server_pkt_mean`
11. `direction_switches`
12. `server_bytes`
13. `size_max`
14. `client_pkt_min`
15. `server_pkt_min`
16. `client_pkt_mean`

### ICS Modbus sensor profile, 16 features

1. `min`
2. `max`
3. `mean`
4. `iat_min_us`
5. `iat_max_us`
6. `mean_iat_us`
7. `range`
8. `slope`
9. `ewma_value`
10. `ewma_deviation`
11. `ewma_relative`
12. `cusum_positive`
13. `cusum_negative_abs`
14. `cusum_max`
15. `cusum_alarm`
16. `trend_consistency`

## Important operational notes

- For DoH and ICS request-response matching, symmetric RSS or flow steering should keep both directions on the same worker queue. The code canonicalizes the 5-tuple, but the NIC still needs to deliver reverse directions to the same queue for per-worker state to be complete.
- The RF microkernel uses static arrays and iterative traversal. The JSON loader in the prototype is intentionally not used at runtime.
- The BNN microkernel packs signs into 64-bit words and masks tail bits, so input and hidden dimensions do not need to be multiples of 64.
- The MLP NEON microkernel has scalar tails, so layer sizes do not need to be multiples of four.
- `make perf` removes logs/timing/contracts from the binary. `make log` keeps async logging and instrumentation.
