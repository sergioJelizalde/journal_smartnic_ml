APP := smartnic_ad
SRC := $(wildcard src/*.c)
OBJ := $(SRC:.c=.o)
CFLAGS += -O3 -g -Wall -Wextra -Wno-unused-parameter -std=gnu11 -Iinclude -Imodels
LDLIBS += -lm

DPDK_CFLAGS := $(shell pkg-config --cflags libdpdk 2>/dev/null)
DPDK_LIBS := $(shell pkg-config --libs libdpdk 2>/dev/null)

ifeq ($(strip $(DPDK_CFLAGS)),)
$(warning pkg-config could not find libdpdk. Set PKG_CONFIG_PATH to your DPDK pkgconfig directory.)
endif

CFLAGS += $(DPDK_CFLAGS)
LDLIBS += $(DPDK_LIBS)

all: $(APP)

$(APP): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

perf: CFLAGS += -DAPP_ENABLE_LOGS=0 -DAPP_ENABLE_TIMING=0 -DAPP_ENABLE_CONTRACTS=0 -DNDEBUG
perf: clean $(APP)

log: CFLAGS += -DAPP_ENABLE_LOGS=1 -DAPP_ENABLE_TIMING=1 -DAPP_ENABLE_CONTRACTS=1
log: clean $(APP)

clean:
	rm -f $(OBJ) $(APP)

.PHONY: all clean perf log
