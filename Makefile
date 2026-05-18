.PHONY: all clean

CC ?= gcc
# build_index runs inside the Docker builder (often QEMU on Apple Silicon),
# so it can't use Haswell-specific codegen. Keep it generic.
# The api binary runs on the test machine (Haswell Mac Mini), so we pin
# the ISA there. Override the api arch via API_ARCH=...
BUILD_CFLAGS = -O2 -Wall -Wextra -std=c11
API_ARCH ?=

# PGO support. PGO=generate adds -fprofile-generate; PGO=use adds -fprofile-use.
# PROFILE_DIR is where .gcda files live between the two builds.
PROFILE_DIR ?= /tmp/pgo
PGO ?=
ifeq ($(PGO),generate)
  PGO_FLAGS = -fprofile-generate=$(PROFILE_DIR) -fprofile-dir=$(PROFILE_DIR)
endif
ifeq ($(PGO),use)
  PGO_FLAGS = -fprofile-use=$(PROFILE_DIR) -fprofile-dir=$(PROFILE_DIR) -fprofile-correction
endif

LB_CFLAGS  = -O3 $(API_ARCH) -flto -fno-plt -fno-stack-protector -Wall -Wextra -std=c11
LB_LDFLAGS = -flto
API_CFLAGS = $(LB_CFLAGS) $(PGO_FLAGS)
API_LDFLAGS = $(LB_LDFLAGS) $(PGO_FLAGS)

all: build_index api lb

build_index: src/build_index.c
	$(CC) $(BUILD_CFLAGS) -o $@ $< -lz

api: src/api.c
	$(CC) $(API_CFLAGS) -o $@ $< $(API_LDFLAGS)

lb: src/lb.c
	$(CC) $(LB_CFLAGS) -o $@ $< $(LB_LDFLAGS)

clean:
	rm -f build_index api lb index.bin
