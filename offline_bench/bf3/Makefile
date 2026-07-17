# Makefile for BlueField-3 (aarch64) offline benchmark.
# Run this ON the BF3 (it compiles native aarch64 + NEON).
#
#   make check      # correctness: scalar vs NEON vs XNNPACK, + scalar vs microkernel
#   make bench      # latency: scalar / neon / xnnpack / generated microkernel
#   make all        # both
#   make MODELS=64_32 check bench   # just one model
#   make clean
#
# XNNPACK must be built/installed for aarch64. Point these at your build:
XNNPACK_DIR   ?= $(HOME)/XNNPACK
XNNPACK_INC   ?= -I$(XNNPACK_DIR)/include
XNNPACK_LIB   ?= -L$(XNNPACK_DIR)/build -lXNNPACK -lpthreadpool -lpthread
# If XNNPACK is not available, `make XNN=0 ...` skips every XNNPACK target.
XNN ?= 1

CC      = gcc
# BF3 cores are Cortex-A78AE. -mcpu gives gcc the real scheduling model
# (FMA pipes/latencies), not just the ISA. Override with CPU=cortex-a78ae
# if `native` misdetects, or CPU=armv8-a+simd to reproduce the old build.
CPU    ?= native
CFLAGS  = -O3 -mcpu=$(CPU) -Wall -std=c11
LDLIBS  = -lm

SRC     = src
WEIGHTS = src/weights
BUILD   = build
GEN     = gen_neon_mlp.py

MODELS ?= 16_8 32_16 64_32 128_64_16 256_128_32

# Layer sizes fed to the microkernel generator: 16 inputs + hidden... + 4 classes.
# MUST match LAYER_SIZES in weights/mlp_<model>.h.
SIZES_16_8        = 16 16 8 4
SIZES_32_16       = 16 32 16 4
SIZES_64_32       = 16 64 32 4
SIZES_128_64_16   = 16 128 64 16 4
SIZES_256_128_32  = 16 256 128 32 4

.PHONY: all check bench clean
all: check bench

# ---------------------------------------------------------------------------
# Per-model staging directory: canonical headers (with WEIGHTS_XNN, 4-class)
# come from src/weights/ -- NOT the stale sklearn src/model_*.h. Staging into
# build/<m>/ also makes the hard-coded  #include "mlp_generated.h"  resolve
# against the freshly generated, shape-matched header.
# ---------------------------------------------------------------------------
define STAGE
	@mkdir -p $(BUILD)/$(1)
	@cp $(WEIGHTS)/mlp_$(1).h        $(BUILD)/$(1)/
	@cp $(WEIGHTS)/feature_stats.h   $(BUILD)/$(1)/
	@cp $(SRC)/mlp_bench.c $(SRC)/mlp_correctness.c $(SRC)/test_correctness.c $(BUILD)/$(1)/
	@python3 $(SRC)/$(GEN) $(SIZES_$(1)) -o $(BUILD)/$(1)/mlp_generated.h >/dev/null
endef

MODEL_HDR = -DMODEL_HEADER='"mlp_$(1).h"'

# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
check:
	@for m in $(MODELS); do $(MAKE) --no-print-directory check-$$m; done

check-%:
	$(call STAGE,$*)
	@echo "==================  CORRECTNESS  [$*]  =================="
	$(CC) $(CFLAGS) -DHAVE_NEON $(if $(filter 1,$(XNN)),-DHAVE_XNNPACK) \
	    $(call MODEL_HDR,$*) $(if $(filter 1,$(XNN)),$(XNNPACK_INC)) \
	    $(BUILD)/$*/mlp_correctness.c -o $(BUILD)/$*/correctness_$* \
	    $(if $(filter 1,$(XNN)),$(XNNPACK_LIB)) $(LDLIBS)
	$(CC) $(CFLAGS) $(call MODEL_HDR,$*) \
	    $(BUILD)/$*/test_correctness.c -o $(BUILD)/$*/micro_correctness_$* $(LDLIBS)
	@echo "-- scalar vs NEON vs XNNPACK --"; ./$(BUILD)/$*/correctness_$*
	@echo "-- scalar vs generated microkernel --"; ./$(BUILD)/$*/micro_correctness_$*

# ---------------------------------------------------------------------------
# Latency (each run writes latencies.csv in the model's build dir)
# ---------------------------------------------------------------------------
bench:
	@for m in $(MODELS); do $(MAKE) --no-print-directory bench-$$m; done

bench-%:
	$(call STAGE,$*)
	@echo "==================  LATENCY  [$*]  =================="
	$(CC) $(CFLAGS) -DUSE_SCALAR    $(call MODEL_HDR,$*) $(BUILD)/$*/mlp_bench.c -o $(BUILD)/$*/bench_scalar_$* $(LDLIBS)
	$(CC) $(CFLAGS) -DUSE_NEON      $(call MODEL_HDR,$*) $(BUILD)/$*/mlp_bench.c -o $(BUILD)/$*/bench_neon_$*   $(LDLIBS)
	$(CC) $(CFLAGS) -DUSE_GENERATED $(call MODEL_HDR,$*) $(BUILD)/$*/mlp_bench.c -o $(BUILD)/$*/bench_gen_$*    $(LDLIBS)
ifeq ($(XNN),1)
	$(CC) $(CFLAGS) -DUSE_XNNPACK   $(call MODEL_HDR,$*) $(XNNPACK_INC) $(BUILD)/$*/mlp_bench.c -o $(BUILD)/$*/bench_xnn_$* $(XNNPACK_LIB) $(LDLIBS)
endif
	@cd $(BUILD)/$* && for k in scalar neon gen $(if $(filter 1,$(XNN)),xnn); do \
	    ./bench_$${k}_$* >/dev/null && mv latencies.csv lat_$${k}_$*.csv && echo "  wrote $(BUILD)/$*/lat_$${k}_$*.csv"; \
	done

clean:
	rm -rf $(BUILD)
