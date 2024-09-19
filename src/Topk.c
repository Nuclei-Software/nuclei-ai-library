#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <float.h>

#include "operators.h"
#include "utils.h"

struct operator_pdata_t {
    uint32_t k;
};

typedef struct {
    int32_t min;
    int32_t min_cnt;
    size_t min_fidx;
} TopkStateI32;

typedef struct {
    float16_t min;
    int32_t min_cnt;
    size_t min_fidx;
} TopkStateF16;

typedef struct {
    float32_t min;
    int32_t min_cnt;
    size_t min_fidx;
} TopkStateF32;

static void Swap_int32(int32_t *a, int32_t *b)
{
    int32_t tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

static void Heapify_int32(int32_t arr[], int len, int idx)
{
    int child = idx * 2 + 1;
    while (child < len) {
        if (child + 1 < len && arr[child + 1] < arr[child]) {
            ++child;
        }

        if (arr[child] < arr[idx]) {
            Swap_int32(&arr[child], &arr[idx]);
            idx = child;
            child = idx * 2 + 1;
        } else {
            break;
        }
    }
}

void Topk_int32(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;

    // assert(pdat->k <= len);

    memcpy(py, px, sizeof(int32_t) * pdat->k);
    for (int i = pdat->k / 2 - 1; i >= 0; --i) {
        Heapify_int32(py, pdat->k, i);
    }
    for (int i = pdat->k; i < x->ndata; ++i) {
        if (py[0] < px[i]) {
            py[0] = px[i];
            Heapify_int32(py, pdat->k, 0);
        }
    }
}

void Topk_int32_rvv(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int32_t *px = (int32_t *)x->datas, *pxx;
    int32_t *py = (int32_t *)y->datas, *pyy;
    size_t vl;
    vint32m8_t vx, candidate;
    vint32m1_t acc;
    vbool4_t vb;
    TopkStateI32 state;

    // assert(pdat->k <= max_k);
    const int32_t max_k = __riscv_vlenb() / sizeof(int32_t) * 8;
    if (pdat->k > max_k) {
        return;
    }

    // init vx
    vl = pdat->k;
    vx = __riscv_vle32_v_i32m8(px, vl);

    // find min
    acc = __riscv_vmv_s_x_i32m1(INT32_MAX, 1);
    acc = __riscv_vredmin_vs_i32m8_i32m1(vx, acc, vl);
    state.min = __riscv_vmv_x_s_i32m1_i32(acc);
    vb = __riscv_vmseq_vx_i32m8_b4(vx, state.min, vl);
    state.min_cnt = __riscv_vcpop_m_b4(vb, vl);
    state.min_fidx = __riscv_vfirst_m_b4(vb, vl);

    for (int i = pdat->k; i < x->ndata; ++i) {
        if (px[i] > state.min) {
            // replace
            candidate = __riscv_vmv_s_x_i32m8(px[i], 1);
            vx = __riscv_vslideup_vx_i32m8_tu(vx, candidate, state.min_fidx, state.min_fidx + 1);
            // update state
            state.min_cnt--;
            if (state.min_cnt == 0) {
                // find min
                acc = __riscv_vmv_s_x_i32m1(INT32_MAX, 1);
                acc = __riscv_vredmin_vs_i32m8_i32m1(vx, acc, vl);
                state.min = __riscv_vmv_x_s_i32m1_i32(acc);
                vb = __riscv_vmseq_vx_i32m8_b4(vx, state.min, vl);
                state.min_cnt = __riscv_vcpop_m_b4(vb, vl);
                state.min_fidx = __riscv_vfirst_m_b4(vb, vl);
            } else {
                // update fidx
                vb = __riscv_vmseq_vx_i32m8_b4(vx, state.min, vl);
                state.min_fidx = __riscv_vfirst_m_b4(vb, vl);
            }
        }
    }
    __riscv_vse32_v_i32m8(py, vx, vl);
}

static void Swap_float16(float16_t *a, float16_t *b)
{
    float16_t tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

static void Heapify_float16(float16_t arr[], int len, int idx)
{
    int child = idx * 2 + 1;
    while (child < len) {
        if (child + 1 < len && arr[child + 1] < arr[child]) {
            ++child;
        }

        if (arr[child] < arr[idx]) {
            Swap_float16(&arr[child], &arr[idx]);
            idx = child;
            child = idx * 2 + 1;
        } else {
            break;
        }
    }
}

void Topk_float16_rvv(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas, *pxx;
    float16_t *py = (float16_t *)y->datas, *pyy;
    size_t vl;
    vfloat16m8_t vx, candidate;
    vfloat16m1_t acc;
    vbool2_t vb;
    TopkStateF16 state;

    // assert(pdat->k <= max_k);
    const int32_t max_k = __riscv_vlenb() / sizeof(float16_t) * 8;
    if (pdat->k > max_k) {
        return;
    }

    // init vx
    vl = pdat->k;
    vx = __riscv_vle16_v_f16m8(px, vl);

    // find min
    acc = __riscv_vfmv_s_f_f16m1(FLT16_MAX, 1);
    acc = __riscv_vfredmin_vs_f16m8_f16m1(vx, acc, vl);
    state.min = __riscv_vfmv_f_s_f16m1_f16(acc);
    vb = __riscv_vmfeq_vf_f16m8_b2(vx, state.min, vl);
    state.min_cnt = __riscv_vcpop_m_b2(vb, vl);
    state.min_fidx = __riscv_vfirst_m_b2(vb, vl);

    for (int i = pdat->k; i < x->ndata; ++i) {
        if (px[i] > state.min) {
            // replace
            candidate = __riscv_vfmv_s_f_f16m8(px[i], 1);
            vx = __riscv_vslideup_vx_f16m8_tu(vx, candidate, state.min_fidx, state.min_fidx + 1);
            // update state
            state.min_cnt--;
            if (state.min_cnt == 0) {
                // find min
                acc = __riscv_vfmv_s_f_f16m1(INT32_MAX, 1);
                acc = __riscv_vfredmin_vs_f16m8_f16m1(vx, acc, vl);
                state.min = __riscv_vfmv_f_s_f16m1_f16(acc);
                vb = __riscv_vmfeq_vf_f16m8_b2(vx, state.min, vl);
                state.min_cnt = __riscv_vcpop_m_b2(vb, vl);
                state.min_fidx = __riscv_vfirst_m_b2(vb, vl);
            } else {
                // update fidx
                vb = __riscv_vmfeq_vf_f16m8_b2(vx, state.min, vl);
                state.min_fidx = __riscv_vfirst_m_b2(vb, vl);
            }
        }
    }
    __riscv_vse16_v_f16m8(py, vx, vl);
}

void Topk_float16(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;

    memcpy(py, px, sizeof(float16_t) * pdat->k);
    for (int i = pdat->k / 2 - 1; i >= 0; --i) {
        Heapify_float16(py, pdat->k, i);
    }
    for (int i = pdat->k; i < x->ndata; ++i) {
        if (py[0] < px[i]) {
            py[0] = px[i];
            Heapify_float16(py, pdat->k, 0);
        }
    }
}

static void Swap_float32(float32_t *a, float32_t *b)
{
    float32_t tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

static void Heapify_float32(float32_t arr[], int len, int idx)
{
    int child = idx * 2 + 1;
    while (child < len) {
        if (child + 1 < len && arr[child + 1] < arr[child]) {
            ++child;
        }

        if (arr[child] < arr[idx]) {
            Swap_float32(&arr[child], &arr[idx]);
            idx = child;
            child = idx * 2 + 1;
        } else {
            break;
        }
    }
}

void Topk_float32_rvv(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas, *pxx;
    float32_t *py = (float32_t *)y->datas, *pyy;
    size_t vl;
    vfloat32m8_t vx, candidate;
    vfloat32m1_t acc;
    vbool4_t vb;
    TopkStateF32 state;

    // assert(pdat->k <= max_k);
    const int32_t max_k = __riscv_vlenb() / sizeof(float32_t) * 8;
    if (pdat->k > max_k) {
        return;
    }

    // init vx
    vl = pdat->k;
    vx = __riscv_vle32_v_f32m8(px, vl);

    // find min
    acc = __riscv_vfmv_s_f_f32m1(FLT_MAX, 1);
    acc = __riscv_vfredmin_vs_f32m8_f32m1(vx, acc, vl);
    state.min = __riscv_vfmv_f_s_f32m1_f32(acc);
    vb = __riscv_vmfeq_vf_f32m8_b4(vx, state.min, vl);
    state.min_cnt = __riscv_vcpop_m_b4(vb, vl);
    state.min_fidx = __riscv_vfirst_m_b4(vb, vl);

    for (int i = pdat->k; i < x->ndata; ++i) {
        if (px[i] > state.min) {
            // replace
            candidate = __riscv_vfmv_s_f_f32m8(px[i], 1);
            vx = __riscv_vslideup_vx_f32m8_tu(vx, candidate, state.min_fidx, state.min_fidx + 1);
            // update state
            state.min_cnt--;
            if (state.min_cnt == 0) {
                // find min
                acc = __riscv_vfmv_s_f_f32m1(INT32_MAX, 1);
                acc = __riscv_vfredmin_vs_f32m8_f32m1(vx, acc, vl);
                state.min = __riscv_vfmv_f_s_f32m1_f32(acc);
                vb = __riscv_vmfeq_vf_f32m8_b4(vx, state.min, vl);
                state.min_cnt = __riscv_vcpop_m_b4(vb, vl);
                state.min_fidx = __riscv_vfirst_m_b4(vb, vl);
            } else {
                // update fidx
                vb = __riscv_vmfeq_vf_f32m8_b4(vx, state.min, vl);
                state.min_fidx = __riscv_vfirst_m_b4(vb, vl);
            }
        }
    }
    __riscv_vse32_v_f32m8(py, vx, vl);
}

void Topk_float32(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;

    memcpy(py, px, sizeof(float32_t) * pdat->k);
    for (int i = pdat->k / 2 - 1; i >= 0; --i) {
        Heapify_float32(py, pdat->k, i);
    }
    for (int i = pdat->k; i < x->ndata; ++i) {
        if (py[0] < px[i]) {
            py[0] = px[i];
            Heapify_float32(py, pdat->k, 0);
        }
    }
}

void *GenerateTopkParam(uint32_t k)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)MALLOC_ASSERT(sizeof(struct operator_pdata_t));
    pdat->k = k;
    return pdat;
}

void FreeTopkParam(void **pdat)
{
    free(*pdat);
    *pdat = NULL;
}