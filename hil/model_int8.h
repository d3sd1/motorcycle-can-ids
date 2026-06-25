/* ============================================================================
 *  model_int8.h  --  Integer-only INT8 dense autoencoder IDS (inference API).
 *
 *  Network:  80 -> 40 -> 20 -> 10 -> 20 -> 40 -> 80
 *            ReLU on the five hidden layers, linear output layer.
 *            ~8400 MACs per forward pass.
 *  Quant:    symmetric per-tensor INT8, int32 accumulate, fixed-point
 *            requantization (multiplier + shift), saturate to int8.
 *
 *  Anomaly score = sum of squared reconstruction error between the input
 *  feature vector and the reconstruction, compared against a learned
 *  threshold.  Everything is integer-only on the hot path.
 *
 *  This is GENUINE int8 inference (int8 weights, int8 activations), NOT a
 *  float simulation.  By default it uses a self-contained int8 fully-connected
 *  kernel that is functionally equivalent to CMSIS-NN arm_fully_connected_s8.
 *  Define USE_CMSIS_NN to call the real ARM kernel instead (see model_int8.c).
 * ========================================================================== */
#ifndef MODEL_INT8_H
#define MODEL_INT8_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Feature-vector dimensionality: 13 features x 6 CAN IDs + 2 = 80. */
#define IDS_FEAT_DIM      80
#define IDS_NUM_FEATURES  13
#define IDS_NUM_CAN_IDS   6
#define IDS_EXTRA_FEATS   2

/* Learned anomaly threshold on the integer sum-of-squared-error (SSE) of the
 * 80-dim reconstruction.  Representative value; the trained threshold comes
 * from the offline software pipeline.  Score > threshold => anomaly. */
#define IDS_SSE_THRESHOLD  120000

/**
 * @brief Run one integer-only INT8 inference of the autoencoder IDS.
 *
 * @param feat80         Input feature vector, 80 int8 values (symmetric, zp=0).
 * @param recon_err_out  Out: integer sum-of-squared reconstruction error (SSE)
 *                       between input and reconstruction (>=0). May be NULL.
 * @return 1 if the sample is flagged anomalous (SSE > IDS_SSE_THRESHOLD),
 *         0 otherwise.
 */
int ids_infer_int8(const int8_t *feat80, int32_t *recon_err_out);

/* ---- Footprint helpers (filled in by model_int8.c via sizeof of the real
 *      static buffers, so the benchmark reports measured, not guessed, bytes). */

/** Total Flash bytes occupied by all int8 weight + int32 bias arrays. */
size_t ids_weights_flash_bytes(void);

/** Static RAM bytes of the activation arena (ping-pong int8 buffers). */
size_t ids_arena_ram_bytes(void);

/** Total integer MACs per forward pass (architecture constant). */
uint32_t ids_total_macs(void);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_INT8_H */
