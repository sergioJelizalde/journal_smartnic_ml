# SPDX-License-Identifier: BSD-3-Clause

APP := smartnic_ad
SRCS-y := $(wildcard src/*.c)
OBJ := $(SRCS-y:.c=.o)

PKGCONF ?= pkg-config

# Check DPDK is installed
ifneq ($(shell $(PKGCONF) --exists libdpdk && echo 0),0)
$(error "no installation of DPDK found. Set PKG_CONFIG_PATH to your DPDK pkgconfig directory.")
endif

PC_FILE := $(shell $(PKGCONF) --path libdpdk 2>/dev/null)

CFLAGS += -O3 -g -Wall -Wextra -Wno-unused-parameter -std=gnu11
CFLAGS += -Iinclude -Imodels
CFLAGS += $(shell $(PKGCONF) --cflags libdpdk)
CFLAGS += -mcpu=cortex-a78ae          # DPU ARM CPU
CFLAGS += -DALLOW_EXPERIMENTAL_API

LDFLAGS_SHARED = $(shell $(PKGCONF) --libs libdpdk) -lm
LDFLAGS_STATIC = $(shell $(PKGCONF) --static --libs libdpdk) -lm

# Default targets
all: shared
.PHONY: shared static perf log clean all

shared: build/$(APP)-shared
	ln -sf $(APP)-shared build/$(APP)

static: build/$(APP)-static
	ln -sf $(APP)-static build/$(APP)

build/$(APP)-shared: $(SRCS-y) Makefile $(PC_FILE) | build
	$(CC) $(CFLAGS) $(SRCS-y) -o $@ $(LDFLAGS_SHARED)

build/$(APP)-static: $(SRCS-y) Makefile $(PC_FILE) | build
	$(CC) $(CFLAGS) $(SRCS-y) -o $@ $(LDFLAGS_STATIC)

# Performance build (no logs, no debug)
perf: CFLAGS += -DAPP_ENABLE_LOGS=0 -DAPP_ENABLE_TIMING=0 -DAPP_ENABLE_CONTRACTS=0 -DNDEBUG
perf: clean shared

# Debug/log build
log: CFLAGS += -DAPP_ENABLE_LOGS=1 -DAPP_ENABLE_TIMING=1 -DAPP_ENABLE_CONTRACTS=1
log: clean shared

build:
	@mkdir -p $@

clean:
	rm -f $(OBJ)
	rm -f build/$(APP) build/$(APP)-shared build/$(APP)-static
	test -d build && rmdir -p build || true