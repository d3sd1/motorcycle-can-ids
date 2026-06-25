# ============================================================================
#  Makefile -- Standalone flashable STM32H7A3ZIT6Q HIL image for the INT8
#              autoencoder IDS inference benchmark (semihosting output).
#
#  `make`        -> builds ids_bench.elf + .bin + .hex + .map and prints size.
#  `make size`   -> re-prints the section sizes of the linked ELF.
#  `make clean`  -> removes all build artifacts.
#
#  The single hard requirement: this links cleanly (zero errors) for cortex-m7
#  and arm-none-eabi-size reports real section sizes.
# ============================================================================

# ---- Toolchain (Arm GNU Toolchain 15.2, installed path on this machine) -----
TOOLCHAIN ?= /c/Program Files/Arm/GNU Toolchain mingw-w64-x86_64-arm-none-eabi/bin
CC        := $(TOOLCHAIN)/arm-none-eabi-gcc
OBJCOPY   := $(TOOLCHAIN)/arm-none-eabi-objcopy
SIZE      := $(TOOLCHAIN)/arm-none-eabi-size

# ---- Target CPU flags (STM32H7 Cortex-M7, double-precision FPU, hard ABI) ---
CPUFLAGS  := -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb

# ---- Device selection -------------------------------------------------------
# The part is the STM32H7A3ZIT6 *Q* (SMPS variant); ST ships only the Q device
# header (stm32h7a3xxq.h) in this CMSIS pack, so we define STM32H7A3xxQ (not the
# plain STM32H7A3xx, whose header is absent). Both share the same Cortex-M7.
DEFS      := -DSTM32H7A3xxQ

# ---- Include paths ----------------------------------------------------------
INCLUDES  := -I. -Icmsis/Include -Icmsis/Device/Include

# ---- Compile flags ----------------------------------------------------------
CFLAGS    := $(CPUFLAGS) -O2 -std=c11 -Wall -Wextra \
             $(DEFS) $(INCLUDES) \
             -ffunction-sections -fdata-sections -fno-common

# ---- Sources ----------------------------------------------------------------
C_SRC     := main.c bench.c model_int8.c system_stm32h7xx.c
ASM_SRC   := startup_stm32h7a3zitxq.s
OBJ       := $(C_SRC:.c=.o) $(ASM_SRC:.s=.o)

# ---- Linker -----------------------------------------------------------------
LDSCRIPT  := STM32H7A3ZITXQ_FLASH.ld
LDFLAGS   := $(CPUFLAGS) -T$(LDSCRIPT) \
             -specs=nano.specs -specs=rdimon.specs \
             -Wl,--gc-sections -Wl,-Map=ids_bench.map,--cref \
             -Wl,--print-memory-usage

TARGET    := ids_bench

.PHONY: all size clean
all: $(TARGET).elf $(TARGET).bin $(TARGET).hex
	@echo "==== arm-none-eabi-size $(TARGET).elf ===="
	"$(SIZE)" $(TARGET).elf

# ---- Link -------------------------------------------------------------------
$(TARGET).elf: $(OBJ) $(LDSCRIPT)
	"$(CC)" $(OBJ) $(LDFLAGS) -o $@

# ---- Objcopy outputs --------------------------------------------------------
$(TARGET).bin: $(TARGET).elf
	"$(OBJCOPY)" -O binary $< $@

$(TARGET).hex: $(TARGET).elf
	"$(OBJCOPY)" -O ihex $< $@

# ---- Compile rules ----------------------------------------------------------
%.o: %.c
	"$(CC)" $(CFLAGS) -c $< -o $@

%.o: %.s
	"$(CC)" $(CPUFLAGS) -c $< -o $@

size: $(TARGET).elf
	"$(SIZE)" $(TARGET).elf

clean:
	rm -f $(OBJ) $(TARGET).elf $(TARGET).bin $(TARGET).hex $(TARGET).map
