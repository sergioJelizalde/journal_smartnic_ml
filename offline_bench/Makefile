MODELS   := 16_8 32_16 64_32 128_64_16 256_128_32
FEAT_HDR := models/feature_stats.h
INC      := -I. -Iinclude

CC_X86   := gcc
CC_ARM   := aarch64-linux-gnu-gcc

X86_FLAGS   := -O3 -march=native
AVX_FLAGS   := -O3 -mavx2 -mfma
ARM_FLAGS   := -O3 -static
NEON_FLAGS  := -O3 -march=armv8-a+simd -static

BIN := bin

.PHONY: all x86 arm clean
all: x86 arm

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
