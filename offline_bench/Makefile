MODELS   := 16_8 32_16 64_32 128_64_16 256_128_32
FEAT_HDR := models/feature_stats.h
INC      := -I. -Iinclude

HOST_ARCH := $(shell uname -m)

# CC_X86 / CC_ARM are for CROSS-compiling from an x86 dev box that has
# aarch64-linux-gnu-gcc installed (targets: `make x86`, `make arm`).
CC_X86   := gcc
CC_ARM   := aarch64-linux-gnu-gcc

# CC_NATIVE is whatever `gcc` actually is on THIS machine -- use this
# (via `make native`) when building directly on the BF3 itself, where
# plain `gcc` already IS the ARM compiler and there's no x86 target to
# cross to. This is what fixes "unrecognized -mavx2" on the BF3: that
# error means `make x86` was run on an aarch64 host by mistake.
CC_NATIVE := gcc

X86_FLAGS   := -O3 -march=native
AVX_FLAGS   := -O3 -mavx2 -mfma
ARM_FLAGS   := -O3 -static
NEON_FLAGS  := -O3 -march=armv8-a+simd -static
# native ARM build: no -static requirement (we're not shipping the binary
# anywhere else), and no cross-sysroot to worry about.
ARM_NATIVE_FLAGS  := -O3
NEON_NATIVE_FLAGS := -O3 -march=armv8-a+simd

BIN := bin

.PHONY: all x86 arm native clean
all: x86 arm

# `make native` -- run this ON the machine you actually want numbers from.
# It auto-detects whether that machine is the BF3 (aarch64) or an x86 box
# and builds only the combos that make sense there.
native:
ifeq ($(HOST_ARCH),$(filter $(HOST_ARCH),aarch64 arm64))
	@echo "Host is $(HOST_ARCH) -- building native ARM scalar + NEON (BF3 A78 cores)"
	$(MAKE) $(foreach m,$(MODELS),$(BIN)/native_arm_scalar_$(m)) \
	        $(foreach m,$(MODELS),$(BIN)/native_arm_neon_$(m))
else
	@echo "Host is $(HOST_ARCH) -- building native x86 scalar + AVX2/FMA"
	$(MAKE) $(foreach m,$(MODELS),$(BIN)/x86_scalar_$(m)) \
	        $(foreach m,$(MODELS),$(BIN)/x86_avx_$(m))
endif

$(BIN)/native_arm_scalar_%: src/kernel_scalar.c src/bench_main.c models/model_%.h $(FEAT_HDR) | $(BIN)
	$(CC_NATIVE) $(ARM_NATIVE_FLAGS) $(INC) \
	  -DMODEL_HEADER_FILE='"models/model_$*.h"' -DFEATURE_HEADER_FILE='"$(FEAT_HDR)"' \
	  src/kernel_scalar.c src/bench_main.c -o $@ -lm

$(BIN)/native_arm_neon_%: src/kernel_neon.c src/bench_main.c models/model_%.h $(FEAT_HDR) | $(BIN)
	$(CC_NATIVE) $(NEON_NATIVE_FLAGS) $(INC) \
	  -DMODEL_HEADER_FILE='"models/model_$*.h"' -DFEATURE_HEADER_FILE='"$(FEAT_HDR)"' \
	  src/kernel_neon.c src/bench_main.c -o $@ -lm

x86: $(foreach m,$(MODELS),$(BIN)/x86_scalar_$(m)) \
     $(foreach m,$(MODELS),$(BIN)/x86_avx_$(m))

arm: $(foreach m,$(MODELS),$(BIN)/arm_scalar_$(m)) \
     $(foreach m,$(MODELS),$(BIN)/arm_neon_$(m))

$(BIN):
	mkdir -p $(BIN)

# ---- x86 scalar ----
$(BIN)/x86_scalar_%: src/kernel_scalar.c src/bench_main.c models/model_%.h $(FEAT_HDR) | $(BIN)
	$(CC_X86) $(X86_FLAGS) $(INC) \
	  -DMODEL_HEADER_FILE='"models/model_$*.h"' -DFEATURE_HEADER_FILE='"$(FEAT_HDR)"' \
	  src/kernel_scalar.c src/bench_main.c -o $@ -lm

# ---- x86 AVX2/FMA ----
$(BIN)/x86_avx_%: src/kernel_avx.c src/bench_main.c models/model_%.h $(FEAT_HDR) | $(BIN)
	$(CC_X86) $(AVX_FLAGS) $(INC) \
	  -DMODEL_HEADER_FILE='"models/model_$*.h"' -DFEATURE_HEADER_FILE='"$(FEAT_HDR)"' \
	  src/kernel_avx.c src/bench_main.c -o $@ -lm

# ---- BF3 ARM scalar (cross-compiled; run natively on-device for real numbers) ----
$(BIN)/arm_scalar_%: src/kernel_scalar.c src/bench_main.c models/model_%.h $(FEAT_HDR) | $(BIN)
	$(CC_ARM) $(ARM_FLAGS) $(INC) \
	  -DMODEL_HEADER_FILE='"models/model_$*.h"' -DFEATURE_HEADER_FILE='"$(FEAT_HDR)"' \
	  src/kernel_scalar.c src/bench_main.c -o $@ -lm

# ---- BF3 ARM NEON (cross-compiled; run natively on-device for real numbers) ----
$(BIN)/arm_neon_%: src/kernel_neon.c src/bench_main.c models/model_%.h $(FEAT_HDR) | $(BIN)
	$(CC_ARM) $(NEON_FLAGS) $(INC) \
	  -DMODEL_HEADER_FILE='"models/model_$*.h"' -DFEATURE_HEADER_FILE='"$(FEAT_HDR)"' \
	  src/kernel_neon.c src/bench_main.c -o $@ -lm

# ---- lightweight correctness checks (safe to run under qemu) ----
check: $(foreach m,$(MODELS),$(BIN)/check_x86_scalar_$(m)) \
       $(foreach m,$(MODELS),$(BIN)/check_x86_avx_$(m)) \
       $(foreach m,$(MODELS),$(BIN)/check_arm_scalar_$(m)) \
       $(foreach m,$(MODELS),$(BIN)/check_arm_neon_$(m))

$(BIN)/check_x86_scalar_%: src/kernel_scalar.c src/correctness_check.c models/model_%.h $(FEAT_HDR) | $(BIN)
	$(CC_X86) $(X86_FLAGS) $(INC) -DMODEL_HEADER_FILE='"models/model_$*.h"' -DFEATURE_HEADER_FILE='"$(FEAT_HDR)"' src/kernel_scalar.c src/correctness_check.c -o $@ -lm

$(BIN)/check_x86_avx_%: src/kernel_avx.c src/correctness_check.c models/model_%.h $(FEAT_HDR) | $(BIN)
	$(CC_X86) $(AVX_FLAGS) $(INC) -DMODEL_HEADER_FILE='"models/model_$*.h"' -DFEATURE_HEADER_FILE='"$(FEAT_HDR)"' src/kernel_avx.c src/correctness_check.c -o $@ -lm

$(BIN)/check_arm_scalar_%: src/kernel_scalar.c src/correctness_check.c models/model_%.h $(FEAT_HDR) | $(BIN)
	$(CC_ARM) $(ARM_FLAGS) $(INC) -DMODEL_HEADER_FILE='"models/model_$*.h"' -DFEATURE_HEADER_FILE='"$(FEAT_HDR)"' src/kernel_scalar.c src/correctness_check.c -o $@ -lm

$(BIN)/check_arm_neon_%: src/kernel_neon.c src/correctness_check.c models/model_%.h $(FEAT_HDR) | $(BIN)
	$(CC_ARM) $(NEON_FLAGS) $(INC) -DMODEL_HEADER_FILE='"models/model_$*.h"' -DFEATURE_HEADER_FILE='"$(FEAT_HDR)"' src/kernel_neon.c src/correctness_check.c -o $@ -lm

clean:
	rm -rf $(BIN) results/bench_results.json
