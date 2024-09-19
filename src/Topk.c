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
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    size_t vl;
    vint32m8_t vx;
    vbool4_t mask;
    unsigned long idx;
    int32_t min;

    vl = pdat->k;
    vx = __riscv_vmv_v_x_i32m8(INT32_MIN, vl);

    for (int i = 0; i < vl; i++) {
        mask = __riscv_vmslt_vx_i32m8_b4(vx, px[i], vl);
        idx = __riscv_vcpop_m_b4(mask, vl);
        vx = __riscv_vslide1down_vx_i32m8(vx, px[i], idx);
    }

    min = __riscv_vmv_x_s_i32m8_i32(vx);

    for (int i = pdat->k; i < x->ndata; ++i) {
        if (px[i] > min) {
            mask = __riscv_vmslt_vx_i32m8_b4(vx, px[i], vl);
            idx = __riscv_vcpop_m_b4(mask, vl);
            vx = __riscv_vslide1down_vx_i32m8(vx, px[i], idx);
            min = __riscv_vmv_x_s_i32m8_i32(vx);
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
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    size_t vl;
    vfloat16m8_t vx;
    vbool2_t mask;
    unsigned long idx;
    float16_t min;

    vl = pdat->k;
    vx = __riscv_vfmv_v_f_f16m8(FLT16_MIN, vl);

    for (int i = 0; i < vl; i++) {
        mask = __riscv_vmflt_vf_f16m8_b2(vx, px[i], vl);
        idx = __riscv_vcpop_m_b2(mask, vl);
        vx = __riscv_vfslide1down_vf_f16m8(vx, px[i], idx);
    }

    min = __riscv_vfmv_f_s_f16m8_f16(vx);

    for (int i = pdat->k; i < x->ndata; ++i) {
        if (px[i] > min) {
            mask = __riscv_vmflt_vf_f16m8_b2(vx, px[i], vl);
            idx = __riscv_vcpop_m_b2(mask, vl);
            vx = __riscv_vfslide1down_vf_f16m8(vx, px[i], idx);
            min = __riscv_vfmv_f_s_f16m8_f16(vx);
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
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    size_t vl;
    vfloat32m8_t vx;
    vbool4_t mask;
    unsigned long idx;
    float32_t min;

    vl = pdat->k;
    vx = __riscv_vfmv_v_f_f32m8(FLT_MIN, vl);

    for (int i = 0; i < vl; i++) {
        mask = __riscv_vmflt_vf_f32m8_b4(vx, px[i], vl);
        idx = __riscv_vcpop_m_b4(mask, vl);
        vx = __riscv_vfslide1down_vf_f32m8(vx, px[i], idx);
    }

    min = __riscv_vfmv_f_s_f32m8_f32(vx);

    for (int i = pdat->k; i < x->ndata; ++i) {
        if (px[i] > min) {
            mask = __riscv_vmflt_vf_f32m8_b4(vx, px[i], vl);
            idx = __riscv_vcpop_m_b4(mask, vl);
            vx = __riscv_vfslide1down_vf_f32m8(vx, px[i], idx);
            min = __riscv_vfmv_f_s_f32m8_f32(vx);
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