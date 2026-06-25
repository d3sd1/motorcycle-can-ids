# Optional: building against the real CMSIS-NN kernel

By **default** this HIL package uses a **self-contained int8 fully-connected
kernel** (in `../model_int8.c`, function `fc_s8`) that is **functionally
equivalent** to ARM's `arm_fully_connected_s8`. The default path has **zero
external dependencies** and is what we recommend for the measurement — the
instruction stream of a symmetric per-tensor int8 MAC + fixed-point requantize
is the same work the CMSIS-NN kernel does, and our requantization arithmetic
reproduces the gemmlowp reference (`SaturatingRoundingDoublingHighMul` +
`RoundingDivideByPOT`) bit-for-bit.

If a reviewer asks you to confirm equivalence on the **real ARM kernel**, you
can switch to it:

## Steps

1. Get CMSIS-NN (Apache-2.0) from the public repository:
   `https://github.com/ARM-software/CMSIS-NN`
   (or the CMSIS-NN pack bundled with STM32CubeIDE / STM32Cube.AI).

2. Add these source files to your build (the minimal set for `s8`
   fully-connected on Cortex-M7):

   ```
   Source/FullyConnectedFunctions/arm_fully_connected_s8.c
   Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c
   Source/NNSupportFunctions/arm_nn_requantize.c        (if present as a .c)
   ```

   and the public headers from `Include/`:
   ```
   Include/arm_nnfunctions.h
   Include/arm_nn_types.h
   Include/arm_nnsupportfunctions.h
   Include/arm_nn_math_types.h
   ```

   Place the headers under `cmsis_nn/Include/` (the Makefile adds
   `-Icmsis_nn/Include` when `USE_CMSIS_NN` is set) and add the `.c` files to
   your CubeIDE project / Makefile source list. CMSIS-NN also needs the CMSIS
   core headers (`core_cm7.h` etc.), which the CubeIDE project already provides.

3. Build with the switch defined:

   - CubeIDE: add `USE_CMSIS_NN` to *Project > Properties > C/C++ Build >
     Settings > MCU GCC Compiler > Preprocessor > Defined symbols*.
   - Makefile: `make check USE_CMSIS_NN=1` (after placing the headers/sources).

4. `model_int8.c` then calls `arm_fully_connected_s8` via the `fc_s8_cmsis`
   wrapper. The numeric result is identical to the default kernel; the latency
   should match within measurement noise (the CMSIS-NN DSP path may be a few
   percent faster on the wider layers thanks to SIMD `__SMLAD`).

## Why the default kernel is sufficient for the paper's numbers

Latency, RAM, and Flash are determined by the **network architecture** and the
**int8 quantization scheme**, not by which equivalent kernel implements the MAC
loop. Both kernels:

- execute the same 8400 int8 MACs with int32 accumulation,
- use the same two-buffer ping-pong activation arena (160 bytes),
- store the same 9240 bytes of int8 weights + int32 biases in Flash.

Reporting the default-kernel numbers is therefore a valid on-silicon
measurement of the INT8 autoencoder IDS. Switching to CMSIS-NN is offered only
so reviewers can independently confirm the equivalence.
