/*
******************************************************************************
**  STM32H7A3ZITxQ FLASH linker script for the standalone HIL benchmark image.
**
**  Derived verbatim from the Motloud datalogger CubeIDE linker script
**  (D:/motloud-v3/firmware/datalogger/STM32H7A3ZITXQ_FLASH.ld) with ONE
**  deliberate change for this standalone build:
**
**    * The original /DISCARD/ block that threw away libc.a, libm.a and
**      libgcc.a has been REMOVED. The CubeIDE project provided its own
**      retargeting + soft-float and discarded the toolchain libs; this
**      standalone image instead links newlib-nano + rdimon (semihosting)
**      and the GCC runtime, so those archives MUST be kept or the link
**      fails to resolve printf/vsnprintf/qsort/_write and FP helpers.
**
**  FLASH 2048K @ 0x08000000, RAM 1024K @ 0x24000000, DTCM, ITCM, etc.
**
**  Target : STMicroelectronics STM32H7A3ZIT6Q (Cortex-M7)
**  Distribution: provided AS-IS, no warranty.
**  Copyright (c) 2025 STMicroelectronics. All rights reserved.
*****************************************************************************
*/

/* Entry Point */
ENTRY(Reset_Handler)

/* Highest address of the user mode stack */
_estack = ORIGIN(RAM) + LENGTH(RAM);    /* end of RAM */
/* Generate a link error if heap and stack don't fit into RAM */
_Min_Heap_Size = 0x2000;      /* required amount of heap  */
_Min_Stack_Size = 0x400; /* required amount of stack */

/* Specify the memory areas */
MEMORY
{
  ITCMRAM (xrw)  : ORIGIN = 0x00000000, LENGTH = 64K
  FLASH (rx)     : ORIGIN = 0x08000000, LENGTH = 2048K
  DTCMRAM1 (xrw) : ORIGIN = 0x20000000, LENGTH = 64K
  DTCMRAM2 (xrw) : ORIGIN = 0x20010000, LENGTH = 64K
  RAM (xrw)      : ORIGIN = 0x24000000, LENGTH = 1024K
  RAM_CD (xrw)   : ORIGIN = 0x30000000, LENGTH = 128K
  RAM_SRD (xrw)  : ORIGIN = 0x38000000, LENGTH = 32K
}

/* Define output sections */
SECTIONS
{
  /* The startup code goes first into FLASH */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector)) /* Startup code */
    . = ALIGN(4);
  } >FLASH

  /* The program code and other data goes into FLASH */
  .text :
  {
    . = ALIGN(4);
    *(.text)           /* .text sections (code) */
    *(.text*)          /* .text* sections (code) */
    *(.glue_7)         /* glue arm to thumb code */
    *(.glue_7t)        /* glue thumb to arm code */
    *(.eh_frame)

    KEEP (*(.init))
    KEEP (*(.fini))

    . = ALIGN(4);
    _etext = .;        /* define a global symbols at end of code */
  } >FLASH

  /* Constant data goes into FLASH */
  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)         /* .rodata sections (constants, strings, etc.) */
    *(.rodata*)        /* .rodata* sections (constants, strings, etc.) */
    . = ALIGN(4);
  } >FLASH

  .ARM.extab (READONLY) : /* The READONLY keyword is only supported in GCC11 and later, remove it if using GCC10 or earlier. */
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } >FLASH
  .ARM (READONLY) : /* The READONLY keyword is only supported in GCC11 and later, remove it if using GCC10 or earlier. */
  {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } >FLASH

  .preinit_array (READONLY) :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } >FLASH

  .init_array (READONLY) :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
  } >FLASH

  .fini_array (READONLY) :
  {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array*))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } >FLASH

  /* used by the startup to initialize data */
  _sidata = LOADADDR(.data);

  /* Initialized data sections goes into RAM, load LMA copy after code */
  .data :
  {
    . = ALIGN(4);
    _sdata = .;        /* create a global symbol at data start */
    *(.data)           /* .data sections */
    *(.data*)          /* .data* sections */
    *(.RamFunc)        /* .RamFunc sections */
    *(.RamFunc*)       /* .RamFunc* sections */

    . = ALIGN(4);
    _edata = .;        /* define a global symbol at data end */
  } >RAM AT> FLASH

  /* Uninitialized data section */
  . = ALIGN(4);
  .bss :
  {
    /* This is used by the startup in order to initialize the .bss section */
    _sbss = .;         /* define a global symbol at bss start */
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)

    . = ALIGN(4);
    _ebss = .;         /* define a global symbol at bss end */
    __bss_end__ = _ebss;
  } >RAM

  /* User_heap_stack section, used to check that there is enough RAM left */
  ._user_heap_stack :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = . + _Min_Stack_Size;
    . = ALIGN(8);
  } >RAM

  /* AXI SRAM section for SDMMC DMA buffers (accessible by SDMMC IDMA) */
  .RAM_D1 (NOLOAD) :
  {
    . = ALIGN(32);
    *(.RAM_D1)
    *(.RAM_D1.*)
    . = ALIGN(32);
  } >RAM

  /* DMA accessible RAM section (CD domain - accessible by DMA1/DMA2) */
  .RAM_CD (NOLOAD) :
  {
    . = ALIGN(4);
    *(.RAM_CD)
    *(.RAM_CD.*)
    . = ALIGN(4);
  } >RAM_CD

  /* NOTE: the original CubeIDE script discarded libc.a/libm.a/libgcc.a here.
     That /DISCARD/ block is intentionally omitted for this standalone build
     so newlib-nano + rdimon (semihosting) + libgcc runtime resolve cleanly. */

  .ARM.attributes 0 : { *(.ARM.attributes) }
}
