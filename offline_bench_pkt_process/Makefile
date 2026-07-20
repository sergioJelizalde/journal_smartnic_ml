# SPDX-License-Identifier: BSD-3-Clause
# tracking_bench — three stateful-tracking DPDK benchmarks (no inference)

APPS = bench_tcp_flow bench_tls_appdata bench_modbus

PKGCONF ?= pkg-config
ifneq ($(shell $(PKGCONF) --exists libdpdk && echo 0),0)
$(error "no installation of DPDK found")
endif

CFLAGS  += -O3 -Wall -Wextra $(shell $(PKGCONF) --cflags libdpdk)
CFLAGS  += -DALLOW_EXPERIMENTAL_API
# BlueField-3 (ARM Cortex-A78AE). Comment out for x86 builds.
#CFLAGS += -mcpu=cortex-a78ae
LDFLAGS_SHARED = $(shell $(PKGCONF) --libs libdpdk) -lm

# Workload knobs (examples):
#   make EXTRA_CFLAGS='-DTCPF_N_PACKETS=6 -DTLSB_N_RECORDS=64 -DMB_WINDOW_SIZE=32'
CFLAGS += $(EXTRA_CFLAGS)

all: $(addprefix build/,$(APPS))

build/%: %.c bench_common.h Makefile | build
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDFLAGS_SHARED)

build:
	@mkdir -p $@

.PHONY: clean
clean:
	rm -rf build
