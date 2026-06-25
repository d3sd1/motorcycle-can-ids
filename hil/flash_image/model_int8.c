/* ============================================================================
 *  model_int8.c  --  Integer-only INT8 dense autoencoder IDS (implementation).
 *
 *  Default path: a self-contained int8 fully-connected kernel (symmetric
 *  per-tensor, int32 accumulate, fixed-point requantize, saturate to int8).
 *  The requantization arithmetic reproduces, bit-for-bit, the gemmlowp /
 *  CMSIS-NN reference used by arm_fully_connected_s8:
 *      requantize(acc) = RoundingDivideByPOT(
 *                            SaturatingRoundingDoublingHighMul(
 *                                acc << left_shift, multiplier),
 *                            right_shift)
 *  so the kernel is functionally equivalent to arm_fully_connected_s8 followed
 *  by the per-layer activation clamp (ReLU is fused as activation_min = 0).
 *
 *  Optional path: define USE_CMSIS_NN to call the real ARM kernel. Add the
 *  CMSIS-NN sources to your project (see cmsis_nn/README.md). The numerical
 *  result is identical; only provided so reviewers can confirm equivalence.
 *
 *  IMPORTANT (validity of the HIL measurement): the weight VALUES in
 *  model_weights.h are representative placeholders. Inference latency, RAM and
 *  Flash footprint are determined solely by the network architecture and the
 *  int8 quantization scheme, NOT by the numeric weight values, because the MAC
 *  count, memory layout and executed instruction stream are identical for any
 *  int8 weight set of these dimensions. Only the detection F1 score depends on
 *  the trained weights, and that is measured offline in software, never here.
 * ========================================================================== */
#include "model_int8.h"
#include "model_weights.h"

#include <string.h>

#ifdef USE_CMSIS_NN
#include "arm_nnfunctions.h"   /* user must add CMSIS-NN to the project */
#endif

/* ---------------------------------------------------------------------------
 *  Per-tensor quantization parameters for each fully-connected layer.
 *
 *  Symmetric quantization (all zero-points = 0). The fixed-point multiplier/
 *  shift map the int32 accumulator back into the int8 activation range; the
 *  exact values are representative (weight-value-independent) and chosen so the
 *  accumulator is scaled into range. ReLU is fused as act_min = 0 on the five
 *  hidden layers; the linear output layer uses the full int8 range.
 * ------------------------------------------------------------------------- */
typedef struct {
    int      in_dim;
    int      out_dim;
    const int8_t  *w;       /* [out_dim][in_dim] */
    const int32_t *b;       /* [out_dim]         */
    int32_t  multiplier;    /* Q0.31 fixed-point multiplier (in [2^30, 2^31)) */
    int32_t  shift;         /* requantization shift (negative = right shift)  */
    int32_t  act_min;       /* output activation clamp min (ReLU => 0)        */
    int32_t  act_max;       /* output activation clamp max                    */
} fc_layer_t;

/* multiplier 0x40000000 == 0.5 in Q0.31; effective scale = 0.5 * 2^shift. */
#define M_HALF  0x40000000

static const fc_layer_t LAYERS[6] = {
    /* in, out, weights, biases,   multiplier, shift, act_min, act_max */
    {  80, 40, ae_w0, ae_b0, M_HALF, -11,   0, 127 },  /* ReLU   */
    {  40, 20, ae_w1, ae_b1, M_HALF, -10,   0, 127 },  /* ReLU   */
    {  20, 10, ae_w2, ae_b2, M_HALF,  -9,   0, 127 },  /* ReLU   */
    {  10, 20, ae_w3, ae_b3, M_HALF,  -9,   0, 127 },  /* ReLU   */
    {  20, 40, ae_w4, ae_b4, M_HALF, -10,   0, 127 },  /* ReLU   */
    {  40, 80, ae_w5, ae_b5, M_HALF, -11,-128, 127 },  /* linear */
};

/* ---- Activation arena: two ping-pong int8 buffers sized to the widest layer
 *      (80). One holds the current layer input, the other the output. This is
 *      the entire dynamic working set of the forward pass. ---------------- */
#define ARENA_WIDTH  IDS_FEAT_DIM
static int8_t s_buf_a[ARENA_WIDTH];
static int8_t s_buf_b[ARENA_WIDTH];

/* ===========================================================================
 *  Fixed-point requantization (gemmlowp / CMSIS-NN equivalent)
 * ========================================================================= */

/* SaturatingRoundingDoublingHighMul: (a*b) >> 31 with rounding and saturation,
 * i.e. the high 32 bits of the doubled 64-bit product. Matches gemmlowp. */
static inline int32_t sat_doubling_high_mul(int32_t a, int32_t b)
{
    if (a == INT32_MIN && b == INT32_MIN) {
        return INT32_MAX;                 /* the single overflow case */
    }
    int64_t ab = (int64_t)a * (int64_t)b;
    int32_t nudge = (ab >= 0) ? (1 << 30) : (1 - (1 << 30));
    int32_t high = (int32_t)((ab + nudge) / (1LL << 31));
    return high;
}

/* RoundingDivideByPOT: round-to-nearest divide by 2^exp (exp >= 0). */
static inline int32_t rounding_divide_by_pot(int32_t x, int32_t exp)
{
    if (exp <= 0) {
        return x;
    }
    int32_t mask = (int32_t)((1LL << exp) - 1);
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
    return (x >> exp) + ((remainder > threshold) ? 1 : 0);
}

/* Full requantize of an int32 accumulator using (multiplier, shift). A positive
 * shift is a left shift applied before the doubling-high-mul; a negative shift
 * is a right shift applied after (rounding). Identical to arm_nn_requantize. */
static inline int32_t requantize(int32_t acc, int32_t multiplier, int32_t shift)
{
    int32_t left_shift  = (shift > 0) ? shift : 0;
    int32_t right_shift = (shift > 0) ? 0 : -shift;
    int32_t val = acc * (int32_t)(1 << left_shift);
    return rounding_divide_by_pot(sat_doubling_high_mul(val, multiplier), right_shift);
}

static inline int32_t clamp_i32(int32_t v, int32_t lo, int32_t hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* ===========================================================================
 *  Self-contained int8 fully-connected kernel
 *  Functionally equivalent to CMSIS-NN arm_fully_connected_s8 with all
 *  zero-points = 0 and the activation clamp [act_min, act_max].
 * ========================================================================= */
__attribute__((unused))
static void fc_s8(const int8_t *input, const fc_layer_t *L, int8_t *output)
{
    for (int o = 0; o < L->out_dim; o++) {
        int32_t acc = L->b[o];                 /* int32 bias pre-load */
        const int8_t *wrow = &L->w[(size_t)o * L->in_dim];
        for (int i = 0; i < L->in_dim; i++) {
            acc += (int32_t)input[i] * (int32_t)wrow[i];   /* zp = 0 */
        }
        int32_t q = requantize(acc, L->multiplier, L->shift);
        output[o] = (int8_t)clamp_i32(q, L->act_min, L->act_max);
    }
}

#ifdef USE_CMSIS_NN
/* Thin wrapper that drives the real arm_fully_connected_s8. The numerical
 * contract (symmetric zp=0, fused activation clamp) is identical to fc_s8. */
static void fc_s8_cmsis(const int8_t *input, const fc_layer_t *L, int8_t *output)
{
    cmsis_nn_context ctx = {0};
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant;
    cmsis_nn_dims in_dims, filter_dims, bias_dims, out_dims;

    fc_params.input_offset  = 0;
    fc_params.filter_offset = 0;
    fc_params.output_offset = 0;
    fc_params.activation.min = L->act_min;
    fc_params.activation.max = L->act_max;

    quant.multiplier = L->multiplier;
    quant.shift      = L->shift;

    in_dims.n = 1;  in_dims.h = 1;  in_dims.w = 1;  in_dims.c = L->in_dim;
    filter_dims.n = L->in_dim; filter_dims.h = 1; filter_dims.w = 1; filter_dims.c = L->out_dim;
    bias_dims.n = 1; bias_dims.h = 1; bias_dims.w = 1; bias_dims.c = L->out_dim;
    out_dims.n = 1; out_dims.h = 1; out_dims.w = 1; out_dims.c = L->out_dim;

    (void)arm_fully_connected_s8(&ctx, &fc_params, &quant,
                                 &in_dims, input,
                                 &filter_dims, L->w,
                                 &bias_dims, L->b,
                                 &out_dims, output);
}
#endif /* USE_CMSIS_NN */

/* ===========================================================================
 *  Public inference entry point
 * ========================================================================= */
int ids_infer_int8(const int8_t *feat80, int32_t *recon_err_out)
{
    /* Ping-pong through the six layers. Input -> A -> B -> A -> B -> A -> B. */
    const int8_t *cur = feat80;
    int8_t *bufs[2] = { s_buf_a, s_buf_b };

    for (int l = 0; l < 6; l++) {
        int8_t *out = bufs[l & 1];
#ifdef USE_CMSIS_NN
        fc_s8_cmsis(cur, &LAYERS[l], out);
#else
        fc_s8(cur, &LAYERS[l], out);
#endif
        cur = out;
    }
    /* After 6 layers the reconstruction (80-dim int8) is in bufs[(6-1)&1] = s_buf_b. */
    const int8_t *recon = cur;

    /* Integer sum-of-squared reconstruction error. Input and reconstruction
     * share the same int8 scale/zero-point (the autoencoder reconstructs its
     * own input space), so the SSE is a valid integer anomaly score. Each
     * squared term <= 255^2 = 65025, times 80 < 5.21e6, well within int32. */
    int32_t sse = 0;
    for (int i = 0; i < IDS_FEAT_DIM; i++) {
        int32_t d = (int32_t)feat80[i] - (int32_t)recon[i];
        sse += d * d;
    }

    if (recon_err_out) {
        *recon_err_out = sse;
    }
    return (sse > IDS_SSE_THRESHOLD) ? 1 : 0;
}

/* ===========================================================================
 *  Footprint helpers -- report MEASURED bytes via sizeof of the real arrays.
 * ========================================================================= */
size_t ids_weights_flash_bytes(void)
{
    return sizeof(ae_w0) + sizeof(ae_b0)
         + sizeof(ae_w1) + sizeof(ae_b1)
         + sizeof(ae_w2) + sizeof(ae_b2)
         + sizeof(ae_w3) + sizeof(ae_b3)
         + sizeof(ae_w4) + sizeof(ae_b4)
         + sizeof(ae_w5) + sizeof(ae_b5);
}

size_t ids_arena_ram_bytes(void)
{
    return sizeof(s_buf_a) + sizeof(s_buf_b);
}

uint32_t ids_total_macs(void)
{
    uint32_t macs = 0;
    for (int l = 0; l < 6; l++) {
        macs += (uint32_t)LAYERS[l].in_dim * (uint32_t)LAYERS[l].out_dim;
    }
    return macs;   /* 8400 */
}
