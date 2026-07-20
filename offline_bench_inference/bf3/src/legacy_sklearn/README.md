# Legacy sklearn headers (unused)

These `model_*.h` and `feature_stats.h` came from an earlier sklearn
`train_multi_mlp.py` pipeline: **binary** (2-class, output size 1) and
**without** the `WEIGHTS_XNN`/`BIASES_XNN` arrays the XNNPACK path needs.

The build uses the canonical torch-generated headers in `../weights/`
(`mlp_*.h`, 4-class, XNN-enabled) instead. Kept here only for reference.
