# ============================================================================
#  Makefile -- STM32H7 Cortex-M7 HIL benchmark for the INT8 autoencoder IDS.
#
#  Two targets matter:
#    make check   -> compile-only (-c) ALL sources with -Werror for cortex-m7.
#                    PROVES the package builds clean for the target. Zero deps.
#    make elf     -> attempts a full link into a standalone .elf/.bin. This
#                    needs the CubeIDE startup file + linker script (see below);
#                    if you don't provide them, use `make check` + the CubeIDE
#                    drop-in route in README_HIL.md (the PRIMARY flashing path).
#
#  The package's primary, supported flashing route is the CubeIDE drop-in
#  described in README_HIL.md. This Makefile is the secondary/standalone route
#  and the CI-style proof that the C compiles warning-free for the H7.
# ============================================================================

# ---- Toolchain -------------------------------------------------------------
# Installed location on this machine (Arm GNU Toolchain 15.2.Rel1):
TOOLCHAIN ?= /c/Program Files/Arm/GNU Toolchain mingw-w64-x86_64-arm-none-eabi/bin
CC        := $(TOOLCHAIN)/arm-none-eabi-gcc
OBJCOPY   := $(TOOLCHAIN)/arm-none-eabi-objcopy
SIZE      := $(TOOLCHAIN)/arm-none-eabi-size

# ---- Target CPU flags (STM32H7 Cortex-M7, double-precision FPU) -------------
CPUFLAGS  := -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard

# ---- Compile flags ---------------------------------------------------------
# -O2 as requested; full warnings + -Werror so `make check` fails on any warning.
CFLAGS    := $(CPUFLAGS) -O2 -std=c11 \
             -Wall -Wextra -Werror \
             -ffunction-sections -fdata-sections \
             -ffreestanding -fno-common \
             -I.

# Define USE_CMSIS_NN=1 on the command line to build against the real ARM kernel
# (you must then add CMSIS-NN sources + include path; see cmsis_nn/README.md).
ifdef USE_CMSIS_NN
CFLAGS    += -DUSE_CMSIS_NN -Icmsis_nn/Include
endif

# ---- Sources ---------------------------------------------------------------
BENCH_SRC := model_int8.c bench.c
APP_SRC   := test_main.c
SRC       := $(BENCH_SRC) $(APP_SRC)
OBJ       := $(SRC:.c=.o)

# Optional CubeIDE artifacts for a real standalone link (provide your own):
STARTUP   ?=
LDSCRIPT  ?=
LDFLAGS   := $(CPUFLAGS) -Wl,--gc-sections -specs=nano.specs -specs=nosys.specs
ifneq ($(LDSCRIPT),)
LDFLAGS   += -T$(LDSCRIPT)
endif

.PHONY: all check elf bin clean

# Default: the compile-check that proves the code is clean for the target.
all: check

# ----------------------------------------------------------------------------
# check : compile every source to an object (no link). This is the guaranteed,
#         dependency-free proof that the C compiles warning-free for cortex-m7.
# ----------------------------------------------------------------------------
check: $(OBJ)
	@echo "OK: all sources compiled clean for cortex-m7 (-Werror)."

%.o: %.c
	"$(CC)" $(CFLAGS) -c $< -o $@

# ----------------------------------------------------------------------------
# elf : full standalone link. Requires STARTUP=<startup_stm32h7a3xx.s> and
#       LDSCRIPT=<STM32H7A3ZITXQ_FLASH.ld> from your CubeIDE project, e.g.:
#
#   make elf \
#     STARTUP=/path/to/Core/Startup/startup_stm32h7a3zitxq.s \
#     LDSCRIPT=/path/to/STM32H7A3ZITXQ_FLASH.ld
#
#       Without them, prefer `make check` + the CubeIDE drop-in route.
# ----------------------------------------------------------------------------
elf: ids_hil.elf
ids_hil.elf: $(OBJ) $(STARTUP)
	@if [ -z "$(LDSCRIPT)" ]; then \
	  echo "ERROR: set LDSCRIPT=<your STM32H7A3ZITXQ_FLASH.ld> (and STARTUP=...) to link."; \
	  echo "       Use 'make check' for the dependency-free compile proof instead."; \
	  exit 1; \
	fi
	"$(CC)" $(CFLAGS) $(OBJ) $(STARTUP) $(LDFLAGS) -o $@
	"$(SIZE)" $@

bin: ids_hil.bin
ids_hil.bin: ids_hil.elf
	"$(OBJCOPY)" -O binary $< $@

clean:
	rm -f $(OBJ) ids_hil.elf ids_hil.bin
