/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri, 2026 ohora23 */
/*
 * quant_dot512.h - the quantized dot products on 512-bit registers
 * (private to quant.c, included once, after quant_dotn.h's AVX-VNNI pass).
 *
 * The same scheme as quant_dotn.h, with the inner loop over 128 weights
 * at a time instead of 64: one zmm holds the nibbles of two 64-weight
 * pieces, and vpmaddubsw + vpdpwssd on zmm do in one instruction what
 * took two on ymm. The integer sums of a block are exact in either width,
 * and the float operations after them are the ones of quant_dotn.h, in
 * the same order: the results are bit-identical to the AVX2, AVX-VNNI,
 * scalar and GPU kernels (test_quant checks it).
 *
 * Needs AVX-512 F, BW, VL and VNNI (Zen 4 and later, Ice Lake and later).
 * The product of one vector is bound by memory on every core count that
 * matters, so what this buys is the multi-vector products of a prompt
 * (bench_kernels' Nx4 and Nx8 columns), where the AVX-VNNI kernels sit
 * below the memory's bandwidth.
 *
 * Only Q4_K so far; the other types fall back to the AVX-VNNI kernels.
 */

#define DOT512_TARGET                                                          \
    __attribute__((target("avx2,fma,f16c,avxvnni,avx512f,avx512bw,avx512vl,avx512vnni")))

/* The low ymm plus the high ymm of v (exact). */
static inline DOT512_TARGET __attribute__((always_inline)) __m256i
fold512_epi32(__m512i v)
{
    return _mm256_add_epi32(_mm512_castsi512_si256(v),
                            _mm512_extracti64x4_epi64(v, 1));
}

/*
 * Q4_K. A block's 128 bytes of nibbles are two zmm loads; in each, the
 * low nibbles are the weights of two 32-weight sub-blocks (2c and 2c+2)
 * and the high nibbles two more (2c+1 and 2c+3), where the activations
 * run in order. The 128-bit lanes are regrouped so that W0 = sub-blocks
 * 2c, 2c+1 (activations 64c .. 64c+63, contiguous) and W1 = sub-blocks
 * 2c+2, 2c+3 (the next 64): the shuffle is on the weights, once per
 * block, not on the activations, once per vector.
 */
static inline DOT512_TARGET __attribute__((always_inline)) void
q4k_dotn_body_avx512(const struct janas_block_q4k *w,
                     const struct janas_block_q8k *x, size_t xs, size_t nb,
                     float *out, const int nv)
{
    const __m512i low4 = _mm512_set1_epi8(0x0f);
    __m128 acc4 = _mm_setzero_ps(), acc4n = _mm_setzero_ps(),
           accm4n = _mm_setzero_ps();
    __m256 acc8 = _mm256_setzero_ps(), accm8 = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; b++) {
        if (nv <= 2)
            prefetch_ahead_vnni(&w[b], sizeof(w[b]));
        uint32_t u[4];
        unpack_scales_u32(w[b].scales, u);
        __m128i sm = _mm_loadu_si128((const __m128i *)u);
        /* the eight scales as int16 in every 128-bit lane */
        __m512i sc16 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(sm));
        __m512i sumi[8], sumh[2];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) sumi[v] =
            _mm512_setzero_si512();
        const int nh = nv <= 2 ? nv : 0;
        _Pragma("GCC unroll 4") for (int v = 0; v < nh; v++) sumh[v] =
            _mm512_setzero_si512();
        _Pragma("GCC unroll 2") for (int c = 0; c < 2; c++)
        {
            /* the bytes of pieces 2c and 2c+1 */
            __m512i q4 =
                _mm512_loadu_si512((const void *)(w[b].qs + 64 * c));
            __m512i lo = _mm512_and_si512(q4, low4);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(q4, 4), low4);
            /* lo = [sub 4c, sub 4c+2], hi = [sub 4c+1, sub 4c+3] (32 bytes
               each); regrouped into W0 = [4c, 4c+1], W1 = [4c+2, 4c+3] */
            __m512i w0 = _mm512_shuffle_i64x2(lo, hi, 0x44);
            __m512i w1 = _mm512_shuffle_i64x2(lo, hi, 0xee);
            /* the scale of each half, as int16, built in the 128-bit lanes
               by pshufb as quant_dotn.h does: lanes 0-1 one sub-block,
               lanes 2-3 the next */
            const int sb = 4 * c; /* first sub-block of this half */
#define SCALE_IDX(k) ((short)(((2 * (k) + 1) << 8) | (2 * (k))))
            __m512i i0 = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi16(SCALE_IDX(sb))),
                _mm256_set1_epi16(SCALE_IDX(sb + 1)), 1);
            __m512i i1 = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_set1_epi16(SCALE_IDX(sb + 2))),
                _mm256_set1_epi16(SCALE_IDX(sb + 3)), 1);
#undef SCALE_IDX
            __m512i s0 = _mm512_shuffle_epi8(sc16, i0);
            __m512i s1 = _mm512_shuffle_epi8(sc16, i1);
            _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++)
            {
                const int8_t *a = x[(size_t)v * xs + b].qs + 128 * c;
                __m512i a0 = _mm512_loadu_si512((const void *)a);
                __m512i a1 = _mm512_loadu_si512((const void *)(a + 64));
                sumi[v] = _mm512_dpwssd_epi32(
                    sumi[v], _mm512_maddubs_epi16(w0, a0), s0);
                if (nv <= 2)
                    sumh[v] = _mm512_dpwssd_epi32(
                        sumh[v], _mm512_maddubs_epi16(w1, a1), s1);
                else
                    sumi[v] = _mm512_dpwssd_epi32(
                        sumi[v], _mm512_maddubs_epi16(w1, a1), s1);
            }
        }
        /* from here on quant_dotn.h's reduction, on the folded sums */
        float dw = _cvtsh_ss(w[b].d), dmw = _cvtsh_ss(w[b].dmin);
        __m128i mins16 = _mm_cvtepu8_epi16(_mm_srli_si128(sm, 8));
        __m128i pm[8];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++)
        {
            pm[v] = _mm_setzero_si128();
            if (v < nv) {
                const struct janas_block_q8k *xb = x + (size_t)v * xs + b;
                pm[v] = _mm_madd_epi16(
                    mins16, _mm_loadu_si128((const __m128i *)xb->bsums2));
            }
        }
        if (nv <= 2) {
            __m256i z = _mm256_setzero_si256();
            __m256i a = _mm256_hadd_epi32(
                fold512_epi32(_mm512_add_epi32(sumi[0], sumh[0])),
                nv == 2 ? fold512_epi32(_mm512_add_epi32(sumi[1], sumh[1]))
                        : z);
            __m128i h = _mm_add_epi32(_mm256_castsi256_si128(a),
                                      _mm256_extracti128_si256(a, 1));
            __m128i t = _mm_hadd_epi32(h, _mm_hadd_epi32(pm[0], pm[1]));
            float d80 = x[b].d, d81 = nv == 2 ? x[xs + b].d : 0.0f;
            __m128 d8 = _mm_setr_ps(d80, d81, d80, d81);
            __m128 dd = _mm_setr_ps(dw, dw, dmw, dmw);
            acc4 = _mm_fmadd_ps(_mm_mul_ps(dd, d8), _mm_cvtepi32_ps(t), acc4);
        } else if (nv <= 4) {
            __m256i t = _mm256_hadd_epi32(
                _mm256_hadd_epi32(fold512_epi32(sumi[0]),
                                  fold512_epi32(sumi[1])),
                _mm256_hadd_epi32(fold512_epi32(sumi[2]),
                                  fold512_epi32(sumi[3])));
            __m128i is4 = _mm_add_epi32(_mm256_castsi256_si128(t),
                                        _mm256_extracti128_si256(t, 1));
            __m128i im4 = _mm_hadd_epi32(_mm_hadd_epi32(pm[0], pm[1]),
                                         _mm_hadd_epi32(pm[2], pm[3]));
            __m128 d8 = _mm_setr_ps(x[b].d, x[xs + b].d, x[2 * xs + b].d,
                                    nv == 4 ? x[3 * xs + b].d : 0.0f);
            acc4n = _mm_fmadd_ps(_mm_mul_ps(_mm_set1_ps(dw), d8),
                                 _mm_cvtepi32_ps(is4), acc4n);
            accm4n = _mm_fmadd_ps(_mm_mul_ps(_mm_set1_ps(dmw), d8),
                                  _mm_cvtepi32_ps(im4), accm4n);
        } else {
            __m256i s01 = _mm256_hadd_epi32(fold512_epi32(sumi[0]),
                                            fold512_epi32(sumi[1]));
            __m256i s23 = _mm256_hadd_epi32(fold512_epi32(sumi[2]),
                                            fold512_epi32(sumi[3]));
            __m256i s45 = _mm256_hadd_epi32(fold512_epi32(sumi[4]),
                                            fold512_epi32(sumi[5]));
            __m256i s67 = _mm256_hadd_epi32(fold512_epi32(sumi[6]),
                                            fold512_epi32(sumi[7]));
            __m256i t0 = _mm256_hadd_epi32(s01, s23);
            __m256i t1 = _mm256_hadd_epi32(s45, s67);
            __m256i is8 =
                _mm256_add_epi32(_mm256_permute2x128_si256(t0, t1, 0x20),
                                 _mm256_permute2x128_si256(t0, t1, 0x31));
            __m128i m03 = _mm_hadd_epi32(_mm_hadd_epi32(pm[0], pm[1]),
                                         _mm_hadd_epi32(pm[2], pm[3]));
            __m128i m47 = _mm_hadd_epi32(_mm_hadd_epi32(pm[4], pm[5]),
                                         _mm_hadd_epi32(pm[6], pm[7]));
            __m256i im8 = _mm256_set_m128i(m47, m03);
#define D8(v) (v < nv ? x[(size_t)(v) * xs + b].d : 0.0f)
            __m256 d8 = _mm256_setr_ps(D8(0), D8(1), D8(2), D8(3), D8(4), D8(5),
                                       D8(6), D8(7));
#undef D8
            acc8 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dw), d8),
                                   _mm256_cvtepi32_ps(is8), acc8);
            accm8 = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(dmw), d8),
                                    _mm256_cvtepi32_ps(im8), accm8);
        }
    }
    if (nv <= 2) {
        float r[4];
        _mm_storeu_ps(r, acc4);
        out[0] = r[0] - r[2];
        if (nv == 2)
            out[1] = r[1] - r[3];
    } else if (nv <= 4) {
        float r[4];
        _mm_storeu_ps(r, _mm_sub_ps(acc4n, accm4n));
        _Pragma("GCC unroll 4") for (int v = 0; v < nv; v++) out[v] = r[v];
    } else {
        float r[8];
        _mm256_storeu_ps(r, _mm256_sub_ps(acc8, accm8));
        _Pragma("GCC unroll 8") for (int v = 0; v < nv; v++) out[v] = r[v];
    }
}

static DOT512_TARGET void q4k_dotn_avx512(const struct janas_block_q4k *w,
                                          const struct janas_block_q8k *x,
                                          size_t xs, int nv, size_t nb,
                                          float *out)
{
    switch (nv) {
    case 8:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 8);
        break;
    case 7:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 7);
        break;
    case 6:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 6);
        break;
    case 5:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 5);
        break;
    case 4:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 4);
        break;
    case 3:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 3);
        break;
    case 2:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 2);
        break;
    default:
        q4k_dotn_body_avx512(w, x, xs, nb, out, 1);
    }
}

static DOT512_TARGET float q4k_dot_avx512(const struct janas_block_q4k *w,
                                          const struct janas_block_q8k *x,
                                          size_t nb)
{
    float r;
    q4k_dotn_body_avx512(w, x, 0, nb, &r, 1);
    return r;
}

static DOT512_TARGET void q4k_rows_avx512(const void *w, size_t row_bytes,
                                          const struct janas_block_q8k *x,
                                          size_t nb, size_t rows, float *y)
{
    for (size_t r = 0; r < rows; r++)
        y[r] = q4k_dot_avx512(
            (const struct janas_block_q4k *)((const char *)w + r * row_bytes),
            x, nb);
}

#undef DOT512_TARGET
