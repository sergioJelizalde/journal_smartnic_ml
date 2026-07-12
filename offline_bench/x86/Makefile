# Makefile for x86 benchmark (scalar, AVX, XNNPACK)
# Usage: make all   (builds all targets)
#        make run   (builds + runs all benchmarks)

CC = gcc
CFLAGS_BASE = -O3 -march=native -Wall -Wextra -std=c11 -fPIC
CFLAGS_SCALAR = $(CFLAGS_BASE)
CFLAGS_AVX = $(CFLAGS_BASE) -mavx2 -mfma
CFLAGS_XNNPACK = $(CFLAGS_BASE) -mavx2

# Paths
SRC_DIR = src
INC_DIR = include
BIN_DIR = bin
RES_DIR = ../results/x86
MODELS_DIR = models

# Include directories
INCLUDES = -I$(INC_DIR) -I$(MODELS_DIR)

# XNNPACK paths (adjust if using different installation)
# Option 1: system-installed
# XNNPACK_CFLAGS = $(shell pkg-config --cflags xnnpack 2>/dev/null || echo "-I/usr/include")
# XNNPACK_LIBS = $(shell pkg-config --libs xnnpack 2>/dev/null || echo "-lxnnpack")
#
# Option 2: vendored (third_party/xnnpack)
XNNPACK_DIR = third_party/xnnpack
XNNPACK_CFLAGS = -I$(XNNPACK_DIR)/include
XNNPACK_LIBS = -L$(XNNPACK_DIR)/lib -lxnnpack -lpthread

# Model sizes to test
MODELS = 16_8 32_16 64_32 128_64_16 256_128_32

# ============================================================================
# Main targets
# ============================================================================

.PHONY: all clean run help check

all: $(addprefix $(BIN_DIR)/bench_x86_scalar_,$(MODELS)) \
     $(addprefix $(BIN_DIR)/bench_x86_avx_,$(MODELS)) \
     $(addprefix $(BIN_DIR)/bench_x86_xnnpack_,$(MODELS)) \
     $(addprefix $(BIN_DIR)/check_x86_scalar_,$(MODELS)) \
     $(addprefix $(BIN_DIR)/check_x86_avx_,$(MODELS))

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(RES_DIR):
	mkdir -p $(RES_DIR)

# ============================================================================
# SCALAR builds (portable reference)
# ============================================================================

$(BIN_DIR)/bench_x86_scalar_%: $(SRC_DIR)/bench_main.c $(SRC_DIR)/kernel_scalar.c $(SRC_DIR)/csv_logger.c $(INC_DIR)/csv_logger.h | $(BIN_DIR) $(RES_DIR)
	$(CC) $(CFLAGS_SCALAR) \
		-DMODEL_SIZE=$* \
		-DKERNEL_SCALAR \
		-DDEVICE_X86 \
		$(INCLUDES) \
		-o $@ $^ -lm

$(BIN_DIR)/check_x86_scalar_%: $(SRC_DIR)/correctness_check.c $(SRC_DIR)/kernel_scalar.c | $(BIN_DIR)
	$(CC) $(CFLAGS_SCALAR) \
		-DMODEL_SIZE=$* \
		-DKERNEL_SCALAR \
		-DDEVICE_X86 \
		$(INCLUDES) \
		-o $@ $^ -lm

# ============================================================================
# AVX builds (x86 optimized)
# ============================================================================

$(BIN_DIR)/bench_x86_avx_%: $(SRC_DIR)/bench_main.c $(SRC_DIR)/kernel_avx.c $(SRC_DIR)/csv_logger.c $(INC_DIR)/csv_logger.h | $(BIN_DIR) $(RES_DIR)
	$(CC) $(CFLAGS_AVX) \
		-DMODEL_SIZE=$* \
		-DKERNEL_AVX \
		-DDEVICE_X86 \
		$(INCLUDES) \
		-o $@ $^ -lm

$(BIN_DIR)/check_x86_avx_%: $(SRC_DIR)/correctness_check.c $(SRC_DIR)/kernel_avx.c | $(BIN_DIR)
	$(CC) $(CFLAGS_AVX) \
		-DMODEL_SIZE=$* \
		-DKERNEL_AVX \
		-DDEVICE_X86 \
		$(INCLUDES) \
		-o $@ $^ -lm

# ============================================================================
# XNNPACK builds (system or vendored library)
# ============================================================================

$(BIN_DIR)/bench_x86_xnnpack_%: $(SRC_DIR)/bench_main.c $(SRC_DIR)/kernel_xnnpack.c $(SRC_DIR)/csv_logger.c $(INC_DIR)/csv_logger.h | $(BIN_DIR) $(RES_DIR)
	$(CC) $(CFLAGS_XNNPACK) \
		-DMODEL_SIZE=$* \
		-DKERNEL_XNNPACK \
		-DDEVICE_X86 \
		$(INCLUDES) $(XNNPACK_CFLAGS) \
		-o $@ $^ $(XNNPACK_LIBS) -lm

# ============================================================================
# Utilities
# ============================================================================

run: all $(RES_DIR)
	bash scripts/run_bench.sh

check: $(addprefix $(BIN_DIR)/check_x86_scalar_,$(MODELS)) \
       $(addprefix $(BIN_DIR)/check_x86_avx_,$(MODELS))
	@echo "Correctness checks:"
	@for m in $(MODELS); do \
		diff <(./$(BIN_DIR)/check_x86_scalar_$$m 2>/dev/null | sort) \
		     <(./$(BIN_DIR)/check_x86_avx_$$m 2>/dev/null | sort) && \
		echo "  ✓ $$m (scalar == avx)" || echo "  ✗ $$m MISMATCH"; \
	done

clean:
	rm -rf $(BIN_DIR) $(RES_DIR)/*.csv

help:
	@echo "x86 Benchmark Makefile"
	@echo "  make all      - Build all targets (scalar, AVX, XNNPACK)"
	@echo "  make run      - Build and run all benchmarks"
	@echo "  make check    - Run correctness checks (scalar vs AVX)"
	@echo "  make clean    - Remove binaries and results"
	@echo ""
	@echo "Model sizes: $(MODELS)"
	@echo "Output: $(RES_DIR)/"
