/**
 * @file Topk.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "operators.h"

struct operator_pdata_t {
    uint32_t k;
};

typedef struct {
    int32_t min;
    int32_t min_cnt;
    size_t min_fidx;
} TopkState;

static void Swap(int32_t *a, int32_t *b)
{
    int32_t tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

static void Heapify(int32_t arr[], int len, int idx)
{
    int child = idx * 2 + 1;
    while (child < len) {
        if (child + 1 < len && arr[child + 1] < arr[child]) {
            ++child;
        }

        if (arr[child] < arr[idx]) {
            Swap(&arr[child], &arr[idx]);
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
        Heapify(py, pdat->k, i);
    }
    for (int i = pdat->k; i < x->ndata; ++i) {
        if (py[0] < px[i]) {
            py[0] = px[i];
            Heapify(py, pdat->k, 0);
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
    TopkState state;
    
    // assert(pdat->k <= maxvl);

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

void *GenerateTopkParam(uint32_t k)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)malloc(sizeof(struct operator_pdata_t));
    pdat->k = k;
    return pdat;
}

void FreeTopkParam(void **pdat)
{
    free(*pdat);
    *pdat = NULL;
}