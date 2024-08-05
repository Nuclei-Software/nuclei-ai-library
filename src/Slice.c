/**
 * @file Slice.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief references:
 *        https://onnx.ai/onnx/operators/onnx__Slice.html
 * @version 0.1
 * @date 2024-07-11
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "operators.h"

struct operator_pdata_t {
    int start[2];
    int end[2];
    int step[2];
};

/**
 * example0:
 * input:
 *   x = [1, 2, 3]
 *       [4, 5, 6]
 *       [7, 8, 9]
 *   axis = 0, start = 0, end = 2, step = 1
 * output:
 *   y = [1, 2, 3]
 *       [4, 5, 6]
 *
 * example1:
 * input:
 *   x = [1, 2, 3]
 *       [4, 5, 6]
 *       [7, 8, 9]
 *   axis = 1, start = 0, end = 2, step = 1
 * output:
 *   y = [1, 2]
 *       [4, 5]
 *       [7, 8]
 *
 * example2:
 * input:
 *   x = [1, 2, 3]
 *       [4, 5, 6]
 *       [7, 8, 9]
 *   axis = 0, start = 0, end = 2, step = 1
 *   axis = 1, start = 0, end = 2, step = 1
 * output:
 *   y = [1, 2]
 *       [4, 5]
 */
void Slice_int8(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)node->priv;
    int num, den;

    if (pdat->end[0] <= 0) {
        pdat->end[0] += x->dims[1];
    }
    if (pdat->end[1] <= 0) {
        pdat->end[1] += x->dims[0];
    }

    for (int row = pdat->start[0]; row < pdat->end[0]; row += pdat->step[0]) {
        for (int col = pdat->start[1]; col < pdat->end[1]; col += pdat->step[1]) {
            *(py++) = px[row * x->dims[0] + col];
        }
    }
    num = pdat->end[1] - pdat->start[1];
    den = pdat->step[1];
    y->dims[0] = (num % den) == 0 ? num / den : num / den + 1;
    num = pdat->end[0] - pdat->start[0];
    den = pdat->step[0];
    y->dims[1] = (num % den) == 0 ? num / den : num / den + 1;
    y->ndata = y->dims[0] * y->dims[1];
}

void Slice_int8_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)node->priv;
    int num, den;
    int8_t *pxx, *pyy;
    size_t avl, vl;
    vint8m8_t vx;

    // adjust end when end <= 0
    if (pdat->end[0] <= 0) {
        pdat->end[0] += x->dims[1];
    }
    if (pdat->end[1] <= 0) {
        pdat->end[1] += x->dims[0];
    }

    // calculate the output tensor size
    num = pdat->end[1] - pdat->start[1];
    den = pdat->step[1];
    y->dims[0] = (num % den) == 0 ? num / den : num / den + 1;
    num = pdat->end[0] - pdat->start[0];
    den = pdat->step[0];
    y->dims[1] = (num % den) == 0 ? num / den : num / den + 1;
    y->ndata = y->dims[0] * y->dims[1];

    for (int row = pdat->start[0]; row < pdat->end[0]; row += pdat->step[0]) {
        pxx = px + row * x->dims[0] + pdat->start[1];
        pyy = py;
        avl = y->dims[0];
        for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vlse8_v_i8m8(pxx, pdat->step[1] * sizeof(int8_t), vl);
            __riscv_vse8_v_i8m8(pyy, vx, vl);
            pxx += vl;
            pyy += vl;
        }
        py += y->dims[0];
    }
}

void Slice_int32(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)node->priv;
    int num, den;

    if (pdat->end[0] <= 0) {
        pdat->end[0] += x->dims[1];
    }
    if (pdat->end[1] <= 0) {
        pdat->end[1] += x->dims[0];
    }

    for (int row = pdat->start[0]; row < pdat->end[0]; row += pdat->step[0]) {
        for (int col = pdat->start[1]; col < pdat->end[1]; col += pdat->step[1]) {
            *(py++) = px[row * x->dims[0] + col];
        }
    }
    num = pdat->end[1] - pdat->start[1];
    den = pdat->step[1];
    y->dims[0] = (num % den) == 0 ? num / den : num / den + 1;
    num = pdat->end[0] - pdat->start[0];
    den = pdat->step[0];
    y->dims[1] = (num % den) == 0 ? num / den : num / den + 1;
    y->ndata = y->dims[0] * y->dims[1];
}

void Slice_int32_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)node->priv;
    int num, den;
    int32_t *pxx, *pyy;
    size_t avl, vl;
    vint32m8_t vx;

    // adjust end when end <= 0
    if (pdat->end[0] <= 0) {
        pdat->end[0] += x->dims[1];
    }
    if (pdat->end[1] <= 0) {
        pdat->end[1] += x->dims[0];
    }

    // calculate the output tensor size
    num = pdat->end[1] - pdat->start[1];
    den = pdat->step[1];
    y->dims[0] = (num % den) == 0 ? num / den : num / den + 1;
    num = pdat->end[0] - pdat->start[0];
    den = pdat->step[0];
    y->dims[1] = (num % den) == 0 ? num / den : num / den + 1;
    y->ndata = y->dims[0] * y->dims[1];

    for (int row = pdat->start[0]; row < pdat->end[0]; row += pdat->step[0]) {
        pxx = px + row * x->dims[0] + pdat->start[1];
        pyy = py;
        avl = y->dims[0];
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vlse32_v_i32m8(pxx, pdat->step[1] * sizeof(int32_t), vl);
            __riscv_vse32_v_i32m8(pyy, vx, vl);
            pxx += vl * pdat->step[1];
            pyy += vl;
        }
        py += y->dims[0];
    }
}

void Slice_float16(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)node->priv;
    int num, den;

    if (pdat->end[0] <= 0) {
        pdat->end[0] += x->dims[1];
    }
    if (pdat->end[1] <= 0) {
        pdat->end[1] += x->dims[0];
    }

    for (int row = pdat->start[0]; row < pdat->end[0]; row += pdat->step[0]) {
        for (int col = pdat->start[1]; col < pdat->end[1]; col += pdat->step[1]) {
            *(py++) = px[row * x->dims[0] + col];
        }
    }
    num = pdat->end[1] - pdat->start[1];
    den = pdat->step[1];
    y->dims[0] = (num % den) == 0 ? num / den : num / den + 1;
    num = pdat->end[0] - pdat->start[0];
    den = pdat->step[0];
    y->dims[1] = (num % den) == 0 ? num / den : num / den + 1;
    y->ndata = y->dims[0] * y->dims[1];
}

void Slice_float16_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)node->priv;
    int num, den;
    float16_t *pxx, *pyy;
    size_t avl, vl;
    vfloat16m8_t vx;

    // adjust end when end <= 0
    if (pdat->end[0] <= 0) {
        pdat->end[0] += x->dims[1];
    }
    if (pdat->end[1] <= 0) {
        pdat->end[1] += x->dims[0];
    }

    // calculate the output tensor size
    num = pdat->end[1] - pdat->start[1];
    den = pdat->step[1];
    y->dims[0] = (num % den) == 0 ? num / den : num / den + 1;
    num = pdat->end[0] - pdat->start[0];
    den = pdat->step[0];
    y->dims[1] = (num % den) == 0 ? num / den : num / den + 1;
    y->ndata = y->dims[0] * y->dims[1];

    for (int row = pdat->start[0]; row < pdat->end[0]; row += pdat->step[0]) {
        pxx = px + row * x->dims[0] + pdat->start[1];
        pyy = py;
        avl = y->dims[0];
        for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vlse16_v_f16m8(pxx, pdat->step[1] * sizeof(float16_t), vl);
            __riscv_vse16_v_f16m8(pyy, vx, vl);
            pxx += vl * pdat->step[1];
            pyy += vl;
        }
        py += y->dims[0];
    }
}

void Slice_float32(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)node->priv;
    int num, den;

    if (pdat->end[0] <= 0) {
        pdat->end[0] += x->dims[1];
    }
    if (pdat->end[1] <= 0) {
        pdat->end[1] += x->dims[0];
    }

    for (int row = pdat->start[0]; row < pdat->end[0]; row += pdat->step[0]) {
        for (int col = pdat->start[1]; col < pdat->end[1]; col += pdat->step[1]) {
            *(py++) = px[row * x->dims[0] + col];
        }
    }
    num = pdat->end[1] - pdat->start[1];
    den = pdat->step[1];
    y->dims[0] = (num % den) == 0 ? num / den : num / den + 1;
    num = pdat->end[0] - pdat->start[0];
    den = pdat->step[0];
    y->dims[1] = (num % den) == 0 ? num / den : num / den + 1;
    y->ndata = y->dims[0] * y->dims[1];
}

void Slice_float32_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)node->priv;
    int num, den;
    float32_t *pxx, *pyy;
    size_t avl, vl;
    vfloat32m8_t vx;

    // adjust end when end <= 0
    if (pdat->end[0] <= 0) {
        pdat->end[0] += x->dims[1];
    }
    if (pdat->end[1] <= 0) {
        pdat->end[1] += x->dims[0];
    }

    // calculate the output tensor size
    num = pdat->end[1] - pdat->start[1];
    den = pdat->step[1];
    y->dims[0] = (num % den) == 0 ? num / den : num / den + 1;
    num = pdat->end[0] - pdat->start[0];
    den = pdat->step[0];
    y->dims[1] = (num % den) == 0 ? num / den : num / den + 1;
    y->ndata = y->dims[0] * y->dims[1];

    for (int row = pdat->start[0]; row < pdat->end[0]; row += pdat->step[0]) {
        pxx = px + row * x->dims[0] + pdat->start[1];
        pyy = py;
        avl = y->dims[0];
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vlse32_v_f32m8(pxx, pdat->step[1] * sizeof(float32_t), vl);
            __riscv_vse32_v_f32m8(pyy, vx, vl);
            pxx += vl * pdat->step[1];
            pyy += vl;
        }
        py += y->dims[0];
    }
}

void *GenerateSliceParam(int naxes, int *axes, int *start, int *end, int *step)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)malloc(sizeof(struct operator_pdata_t));
    pdat->start[0] = 0;
    pdat->start[1] = 0;
    pdat->end[0] = 0;
    pdat->end[1] = 0;
    pdat->step[0] = 1;
    pdat->step[1] = 1;
    for (int i = 0; i < naxes; i++) {
        if (axes[i] == 0 || axes[i] == 1) {
            pdat->start[axes[i]] = start[i];
            pdat->end[axes[i]] = end[i];
            pdat->step[axes[i]] = step[i];
        }
    }
    return pdat;
}

void FreeSliceParam(void **pdat)
{
    free(*pdat);
    *pdat = NULL;
}