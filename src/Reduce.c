/*
 * https://www.tensorflow.org/api_docs/python/tf/math
 */

#include <float.h>
#include <stdbool.h>
#include <stdio.h>

#include "operators.h"

void ReduceAll(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    bool *px = (bool *)x->datas;
    bool *py = (bool *)y->datas;
    bool *pxx, *pyy;
    int *paxis = n->priv;
    bool res = true;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res = res && *px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = true;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res = res && *(pxx + k * stride);
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceAll_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    uint8_t *px = (uint8_t *)x->datas, *pxx, *pxxx;
    bool *py = (bool *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vuint8m1_t acc;
    vuint8m8_t vx;
    size_t avl, vl;

    if (paxis == NULL) {
        // reduce all axes
        acc = __riscv_vmv_s_x_u8m1(true, 1);
        avl = x->ndata;
        for (; (vl = __riscv_vsetvl_e8m1(avl)) > 0; avl -= vl) {
            vx = __riscv_vle8_v_u8m8(px, vl);
            px += vl;
            acc = __riscv_vredand_vs_u8m8_u8m1(vx, acc, vl);
        }
        py[0] = __riscv_vmv_x_s_u8m1_u8(acc);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                acc = __riscv_vmv_s_x_u8m1(true, 1);
                pxxx = pxx;
                for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vlse8_v_u8m8(pxxx, stride, vl);
                    pxxx += vl * stride;
                    acc = __riscv_vredand_vs_u8m8_u8m1(vx, acc, vl);
                }
                pxx++;
                *pyy++ = __riscv_vmv_x_s_u8m1_u8(acc);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceAny(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    bool *px = (bool *)x->datas;
    bool *py = (bool *)y->datas;
    bool *pxx, *pyy;
    int *paxis = n->priv;
    bool res = false;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res = res || *px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = false;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res = res || *(pxx + k * stride);
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceAny_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    uint8_t *px = (uint8_t *)x->datas, *pxx, *pxxx;
    bool *py = (bool *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vuint8m1_t acc;
    vuint8m8_t vx;
    size_t avl, vl;

    if (paxis == NULL) {
        // reduce all axes
        acc = __riscv_vmv_s_x_u8m1(false, 1);
        avl = x->ndata;
        for (; (vl = __riscv_vsetvl_e8m1(avl)) > 0; avl -= vl) {
            vx = __riscv_vle8_v_u8m8(px, vl);
            px += vl;
            acc = __riscv_vredor_vs_u8m8_u8m1(vx, acc, vl);
        }
        py[0] = __riscv_vmv_x_s_u8m1_u8(acc);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                acc = __riscv_vmv_s_x_u8m1(false, 1);
                pxxx = pxx;
                for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vlse8_v_u8m8(pxxx, stride, vl);
                    pxxx += vl * stride;
                    acc = __riscv_vredor_vs_u8m8_u8m1(vx, acc, vl);
                }
                pxx++;
                *pyy++ = __riscv_vmv_x_s_u8m1_u8(acc);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMax_int8(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    int8_t *pxx, *pyy;
    int *paxis = n->priv;
    int8_t res = INT8_MIN;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res = *px > res ? *px : res;
            px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = INT8_MIN;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res = *(pxx + k * stride) > res ? *(pxx + k * stride) : res;
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMax_int8_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int8_t *px = (int8_t *)x->datas, *pxx, *pxxx;
    int8_t *py = (int8_t *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vint8m1_t acc;
    vint8m8_t vx;
    size_t avl, vl;

    if (paxis == NULL) {
        // reduce all axes
        acc = __riscv_vmv_s_x_i8m1(INT8_MIN, 1);
        avl = x->ndata;
        for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle8_v_i8m8(px, vl);
            px += vl;
            acc = __riscv_vredmax_vs_i8m8_i8m1(vx, acc, vl);
        }
        py[0] = __riscv_vmv_x_s_i8m1_i8(acc);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                acc = __riscv_vmv_s_x_i8m1(INT8_MIN, 1);
                pxxx = pxx;
                for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vlse8_v_i8m8(pxxx, stride * sizeof(int8_t), vl);
                    pxxx += vl * stride;
                    acc = __riscv_vredmax_vs_i8m8_i8m1(vx, acc, vl);
                }
                pxx++;
                *pyy++ = __riscv_vmv_x_s_i8m1_i8(acc);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMax_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    float16_t *pxx, *pyy;
    int *paxis = n->priv;
    float16_t res = FLT_MIN;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res = *px > res ? *px : res;
            px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = FLT_MIN;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res = *(pxx + k * stride) > res ? *(pxx + k * stride) : res;
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMax_float16_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas, *pxx, *pxxx;
    float16_t *py = (float16_t *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vfloat16m1_t acc;
    vfloat16m8_t vx;
    size_t avl, vl;

    if (paxis == NULL) {
        // reduce all axes
        acc = __riscv_vfmv_s_f_f16m1(FLT_MIN, 1);
        avl = x->ndata;
        for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle16_v_f16m8(px, vl);
            px += vl;
            acc = __riscv_vfredmax_vs_f16m8_f16m1(vx, acc, vl);
        }
        py[0] = __riscv_vfmv_f_s_f16m1_f16(acc);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                acc = __riscv_vfmv_s_f_f16m1(FLT_MIN, 1);
                pxxx = pxx;
                for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vlse16_v_f16m8(pxxx, stride * sizeof(float16_t), vl);
                    pxxx += vl * stride;
                    acc = __riscv_vfredmax_vs_f16m8_f16m1(vx, acc, vl);
                }
                pxx++;
                *pyy++ = __riscv_vfmv_f_s_f16m1_f16(acc);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMax_int32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    int32_t *pxx, *pyy;
    int *paxis = n->priv;
    int32_t res = FLT_MIN;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res = *px > res ? *px : res;
            px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = FLT_MIN;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res = *(pxx + k * stride) > res ? *(pxx + k * stride) : res;
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMax_int32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int32_t *px = (int32_t *)x->datas, *pxx, *pxxx;
    int32_t *py = (int32_t *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vint32m1_t acc;
    vint32m8_t vx;
    size_t avl, vl;

    if (paxis == NULL) {
        // reduce all axes
        acc = __riscv_vmv_s_x_i32m1(INT32_MIN, 1);
        avl = x->ndata;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle32_v_i32m8(px, vl);
            px += vl;
            acc = __riscv_vredmax_vs_i32m8_i32m1(vx, acc, vl);
        }
        py[0] = __riscv_vmv_x_s_i32m1_i32(acc);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                acc = __riscv_vmv_s_x_i32m1(INT32_MIN, 1);
                pxxx = pxx;
                for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vlse32_v_i32m8(pxxx, stride * sizeof(int32_t), vl);
                    pxxx += vl * stride;
                    acc = __riscv_vredmax_vs_i32m8_i32m1(vx, acc, vl);
                }
                pxx++;
                *pyy++ = __riscv_vmv_x_s_i32m1_i32(acc);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMax_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    float32_t *pxx, *pyy;
    int *paxis = n->priv;
    float32_t res = FLT_MIN;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res = *px > res ? *px : res;
            px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = FLT_MIN;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res = *(pxx + k * stride) > res ? *(pxx + k * stride) : res;
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMax_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas, *pxx, *pxxx;
    float32_t *py = (float32_t *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vfloat32m1_t acc;
    vfloat32m8_t vx;
    size_t avl, vl;

    if (paxis == NULL) {
        // reduce all axes
        acc = __riscv_vfmv_s_f_f32m1(FLT_MIN, 1);
        avl = x->ndata;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle32_v_f32m8(px, vl);
            px += vl;
            acc = __riscv_vfredmax_vs_f32m8_f32m1(vx, acc, vl);
        }
        py[0] = __riscv_vfmv_f_s_f32m1_f32(acc);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                acc = __riscv_vfmv_s_f_f32m1(FLT_MIN, 1);
                pxxx = pxx;
                for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vlse32_v_f32m8(pxxx, stride * sizeof(float32_t), vl);
                    pxxx += vl * stride;
                    acc = __riscv_vfredmax_vs_f32m8_f32m1(vx, acc, vl);
                }
                pxx++;
                *pyy++ = __riscv_vfmv_f_s_f32m1_f32(acc);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMin_int8(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    int8_t *pxx, *pyy;
    int *paxis = n->priv;
    int8_t res = INT8_MAX;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res = *px < res ? *px : res;
            px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = INT8_MAX;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res = *(pxx + k * stride) < res ? *(pxx + k * stride) : res;
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMin_int8_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int8_t *px = (int8_t *)x->datas, *pxx, *pxxx;
    int8_t *py = (int8_t *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vint8m1_t acc;
    vint8m8_t vx;
    size_t avl, vl;

    if (paxis == NULL) {
        // reduce all axes
        acc = __riscv_vmv_s_x_i8m1(INT8_MAX, 1);
        avl = x->ndata;
        for (; (vl = __riscv_vsetvl_e8m1(avl)) > 0; avl -= vl) {
            vx = __riscv_vle8_v_i8m8(px, vl);
            px += vl;
            acc = __riscv_vredmin_vs_i8m8_i8m1(vx, acc, vl);
        }
        py[0] = __riscv_vmv_x_s_i8m1_i8(acc);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                acc = __riscv_vmv_s_x_i8m1(INT8_MAX, 1);
                pxxx = pxx;
                for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vlse8_v_i8m8(pxxx, stride, vl);
                    pxxx += vl * stride;
                    acc = __riscv_vredmin_vs_i8m8_i8m1(vx, acc, vl);
                }
                pxx++;
                *pyy++ = __riscv_vmv_x_s_i8m1_i8(acc);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMin_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    float16_t *pxx, *pyy;
    int *paxis = n->priv;
    float16_t res = FLT_MAX;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res = *px < res ? *px : res;
            px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = FLT_MAX;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res = *(pxx + k * stride) < res ? *(pxx + k * stride) : res;
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMin_float16_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas, *pxx, *pxxx;
    float16_t *py = (float16_t *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vfloat16m1_t acc;
    vfloat16m8_t vx;
    size_t avl, vl;

    if (paxis == NULL) {
        // reduce all axes
        acc = __riscv_vfmv_s_f_f16m1(FLT_MAX, 1);
        avl = x->ndata;
        for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle16_v_f16m8(px, vl);
            px += vl;
            acc = __riscv_vfredmin_vs_f16m8_f16m1(vx, acc, vl);
        }
        py[0] = __riscv_vfmv_f_s_f16m1_f16(acc);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                acc = __riscv_vfmv_s_f_f16m1(FLT_MAX, 1);
                pxxx = pxx;
                for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vlse16_v_f16m8(pxxx, stride * sizeof(float16_t), vl);
                    pxxx += vl * stride;
                    acc = __riscv_vfredmin_vs_f16m8_f16m1(vx, acc, vl);
                }
                pxx++;
                *pyy++ = __riscv_vfmv_f_s_f16m1_f16(acc);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMin_int32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    int32_t *pxx, *pyy;
    int *paxis = n->priv;
    int32_t res = INT32_MAX;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res = *px < res ? *px : res;
            px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = INT32_MAX;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res = *(pxx + k * stride) < res ? *(pxx + k * stride) : res;
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMin_int32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int32_t *px = (int32_t *)x->datas, *pxx, *pxxx;
    int32_t *py = (int32_t *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vint32m1_t acc;
    vint32m8_t vx;
    size_t avl, vl;

    if (paxis == NULL) {
        // reduce all axes
        acc = __riscv_vmv_s_x_i32m1(INT32_MAX, 1);
        avl = x->ndata;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle32_v_i32m8(px, vl);
            px += vl;
            acc = __riscv_vredmin_vs_i32m8_i32m1(vx, acc, vl);
        }
        py[0] = __riscv_vmv_x_s_i32m1_i32(acc);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                acc = __riscv_vmv_s_x_i32m1(INT32_MAX, 1);
                pxxx = pxx;
                for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vlse32_v_i32m8(pxxx, stride * sizeof(int32_t), vl);
                    pxxx += vl * stride;
                    acc = __riscv_vredmin_vs_i32m8_i32m1(vx, acc, vl);
                }
                pxx++;
                *pyy++ = __riscv_vmv_x_s_i32m1_i32(acc);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMin_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    float32_t *pxx, *pyy;
    int *paxis = n->priv;
    float32_t res = FLT_MAX;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res = *px < res ? *px : res;
            px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = FLT_MAX;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res = *(pxx + k * stride) < res ? *(pxx + k * stride) : res;
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceMin_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas, *pxx, *pxxx;
    float32_t *py = (float32_t *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vfloat32m1_t acc;
    vfloat32m8_t vx;
    size_t avl, vl;

    if (paxis == NULL) {
        // reduce all axes
        acc = __riscv_vfmv_s_f_f32m1(FLT_MAX, 1);
        avl = x->ndata;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle32_v_f32m8(px, vl);
            px += vl;
            acc = __riscv_vfredmin_vs_f32m8_f32m1(vx, acc, vl);
        }
        py[0] = __riscv_vfmv_f_s_f32m1_f32(acc);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                acc = __riscv_vfmv_s_f_f32m1(FLT_MAX, 1);
                pxxx = pxx;
                for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vlse32_v_f32m8(pxxx, stride * sizeof(float32_t), vl);
                    pxxx += vl * stride;
                    acc = __riscv_vfredmin_vs_f32m8_f32m1(vx, acc, vl);
                }
                pxx++;
                *pyy++ = __riscv_vfmv_f_s_f32m1_f32(acc);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceProd_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    float16_t *pxx, *pyy;
    int *paxis = n->priv;
    float16_t res = 1.0f;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res *= *px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = 1.0f;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res *= *(pxx + k * stride);
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceProd_float16_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas, *pxx, *pxxx;
    float16_t *py = (float16_t *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vfloat16m1_t acc;
    vfloat16m8_t vx, vxx;
    size_t avl, vl, maxvl, offset;
    vfloat16m4_t a, b;
    vfloat16m2_t c, d;
    vfloat16m1_t e, f;

    if (paxis == NULL) {
        // reduce all axes
        maxvl = __riscv_vsetvlmax_e16m8();
        avl = x->ndata;
        if (avl >= maxvl) {
            vx = __riscv_vle16_v_f16m8(px, maxvl);
            px += maxvl;
            avl -= maxvl;
            // load remained data, and multiply with vx
            for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
                if (vl < maxvl) {
                    vxx = __riscv_vfmv_v_f_f16m8(1.0f, maxvl);
                    vxx = __riscv_vle16_v_f16m8_tu(vxx, px, vl);
                } else {
                    vxx = __riscv_vle16_v_f16m8(px, vl);
                }
                vx = __riscv_vfmul_vv_f16m8(vx, vxx, vl);
                px += vl;
            }
        } else {
            // load data to vx
            vx = __riscv_vfmv_v_f_f16m8(1.0f, maxvl);
            vx = __riscv_vle16_v_f16m8_tu(vx, px, avl);
        }
        a = __riscv_vget_v_f16m8_f16m4(vx, 0);
        b = __riscv_vget_v_f16m8_f16m4(vx, 1);
        vl = __riscv_vsetvlmax_e16m4();
        a = __riscv_vfmul_vv_f16m4(a, b, vl);

        c = __riscv_vget_v_f16m4_f16m2(a, 0);
        d = __riscv_vget_v_f16m4_f16m2(a, 1);
        vl = __riscv_vsetvlmax_e16m2();
        c = __riscv_vfmul_vv_f16m2(c, d, vl);

        e = __riscv_vget_v_f16m2_f16m1(c, 0);
        f = __riscv_vget_v_f16m2_f16m1(c, 1);
        vl = __riscv_vsetvlmax_e16m1();
        e = __riscv_vfmul_vv_f16m1(e, f, vl);

        vl >>= 1;
        for (; vl >= 1; vl >>= 1) {
            f = __riscv_vslidedown_vx_f16m1(e, vl, vl);
            e = __riscv_vfmul_vv_f16m1(e, f, vl);
        }
        py[0] = __riscv_vfmv_f_s_f16m1_f16(e);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                pxxx = pxx;
                maxvl = __riscv_vsetvlmax_e16m8();
                if (avl >= maxvl) {
                    vx = __riscv_vlse16_v_f16m8(pxxx, sizeof(float16_t) * stride, maxvl);
                    px += maxvl;
                    avl -= maxvl;
                    // load remained data, and multiply with vx
                    for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
                        if (vl < maxvl) {
                            vxx = __riscv_vfmv_v_f_f16m8(1.0f, maxvl);
                            vxx = __riscv_vlse16_v_f16m8_tu(vxx, pxxx, sizeof(float16_t) * stride, vl);
                        } else {
                            vxx = __riscv_vlse16_v_f16m8(pxxx, sizeof(float16_t) * stride, vl);
                        }
                        pxxx += vl * stride;
                        vx = __riscv_vfmul_vv_f16m8(vx, vxx, vl);
                    }
                } else {
                    // load data to vx
                    vx = __riscv_vfmv_v_f_f16m8(1.0f, maxvl);
                    vx = __riscv_vlse16_v_f16m8_tu(vx, pxxx, sizeof(float16_t) * stride, avl);
                }

                a = __riscv_vget_v_f16m8_f16m4(vx, 0);
                b = __riscv_vget_v_f16m8_f16m4(vx, 1);
                vl = __riscv_vsetvlmax_e16m4();
                a = __riscv_vfmul_vv_f16m4(a, b, vl);

                c = __riscv_vget_v_f16m4_f16m2(a, 0);
                d = __riscv_vget_v_f16m4_f16m2(a, 1);
                vl = __riscv_vsetvlmax_e16m2();
                c = __riscv_vfmul_vv_f16m2(c, d, vl);

                e = __riscv_vget_v_f16m2_f16m1(c, 0);
                f = __riscv_vget_v_f16m2_f16m1(c, 1);
                vl = __riscv_vsetvlmax_e16m1();
                e = __riscv_vfmul_vv_f16m1(e, f, vl);

                vl >>= 1;
                for (; vl >= 1; vl >>= 1) {
                    f = __riscv_vslidedown_vx_f16m1(e, vl, vl);
                    e = __riscv_vfmul_vv_f16m1(e, f, vl);
                }
                pxx++;
                *pyy++ = __riscv_vfmv_f_s_f16m1_f16(e);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceProd_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    float32_t *pxx, *pyy;
    int *paxis = n->priv;
    float32_t res = 1.0f;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res *= *px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = 1.0f;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res *= *(pxx + k * stride);
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceProd_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas, *pxx, *pxxx;
    float32_t *py = (float32_t *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vfloat32m1_t acc;
    vfloat32m8_t vx, vxx;
    size_t avl, vl, maxvl, offset;
    vfloat32m4_t a, b;
    vfloat32m2_t c, d;
    vfloat32m1_t e, f;

    if (paxis == NULL) {
        // reduce all axes
        maxvl = __riscv_vsetvlmax_e32m8();
        avl = x->ndata;
        if (avl >= maxvl) {
            vx = __riscv_vle32_v_f32m8(px, maxvl);
            px += maxvl;
            avl -= maxvl;
            // load remained data, and multiply with vx
            for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                if (vl < maxvl) {
                    vxx = __riscv_vfmv_v_f_f32m8(1.0f, maxvl);
                    vxx = __riscv_vle32_v_f32m8_tu(vxx, px, vl);
                } else {
                    vxx = __riscv_vle32_v_f32m8(px, vl);
                }
                vx = __riscv_vfmul_vv_f32m8(vx, vxx, vl);
                px += vl;
            }
        } else {
            // load data to vx
            vx = __riscv_vfmv_v_f_f32m8(1.0f, maxvl);
            vx = __riscv_vle32_v_f32m8_tu(vx, px, avl);
        }
        a = __riscv_vget_v_f32m8_f32m4(vx, 0);
        b = __riscv_vget_v_f32m8_f32m4(vx, 1);
        vl = __riscv_vsetvlmax_e32m4();
        a = __riscv_vfmul_vv_f32m4(a, b, vl);

        c = __riscv_vget_v_f32m4_f32m2(a, 0);
        d = __riscv_vget_v_f32m4_f32m2(a, 1);
        vl = __riscv_vsetvlmax_e32m2();
        c = __riscv_vfmul_vv_f32m2(c, d, vl);

        e = __riscv_vget_v_f32m2_f32m1(c, 0);
        f = __riscv_vget_v_f32m2_f32m1(c, 1);
        vl = __riscv_vsetvlmax_e32m1();
        e = __riscv_vfmul_vv_f32m1(e, f, vl);

        vl >>= 1;
        for (; vl >= 1; vl >>= 1) {
            f = __riscv_vslidedown_vx_f32m1(e, vl, vl);
            e = __riscv_vfmul_vv_f32m1(e, f, vl);
        }
        py[0] = __riscv_vfmv_f_s_f32m1_f32(e);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                pxxx = pxx;
                maxvl = __riscv_vsetvlmax_e32m8();
                if (avl >= maxvl) {
                    vx = __riscv_vlse32_v_f32m8(pxxx, sizeof(float32_t) * stride, maxvl);
                    px += maxvl;
                    avl -= maxvl;
                    // load remained data, and multiply with vx
                    for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                        if (vl < maxvl) {
                            vxx = __riscv_vfmv_v_f_f32m8(1.0f, maxvl);
                            vxx = __riscv_vlse32_v_f32m8_tu(vxx, pxxx, sizeof(float32_t) * stride, vl);
                        } else {
                            vxx = __riscv_vlse32_v_f32m8(pxxx, sizeof(float32_t) * stride, vl);
                        }
                        pxxx += vl * stride;
                        vx = __riscv_vfmul_vv_f32m8(vx, vxx, vl);
                    }
                } else {
                    // load data to vx
                    vx = __riscv_vfmv_v_f_f32m8(1.0f, maxvl);
                    vx = __riscv_vlse32_v_f32m8_tu(vx, pxxx, sizeof(float32_t) * stride, avl);
                }

                a = __riscv_vget_v_f32m8_f32m4(vx, 0);
                b = __riscv_vget_v_f32m8_f32m4(vx, 1);
                vl = __riscv_vsetvlmax_e32m4();
                a = __riscv_vfmul_vv_f32m4(a, b, vl);

                c = __riscv_vget_v_f32m4_f32m2(a, 0);
                d = __riscv_vget_v_f32m4_f32m2(a, 1);
                vl = __riscv_vsetvlmax_e32m2();
                c = __riscv_vfmul_vv_f32m2(c, d, vl);

                e = __riscv_vget_v_f32m2_f32m1(c, 0);
                f = __riscv_vget_v_f32m2_f32m1(c, 1);
                vl = __riscv_vsetvlmax_e32m1();
                e = __riscv_vfmul_vv_f32m1(e, f, vl);

                vl >>= 1;
                for (; vl >= 1; vl >>= 1) {
                    f = __riscv_vslidedown_vx_f32m1(e, vl, vl);
                    e = __riscv_vfmul_vv_f32m1(e, f, vl);
                }
                pxx++;
                *pyy++ = __riscv_vfmv_f_s_f32m1_f32(e);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceSum_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    float16_t *pxx, *pyy;
    int *paxis = n->priv;
    float16_t res = 0.0f;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res += *px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = 0.0f;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res += *(pxx + k * stride);
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceSum_float16_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas, *pxx, *pxxx;
    float16_t *py = (float16_t *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vfloat16m1_t acc;
    vfloat16m8_t vx;
    size_t avl, vl;

    if (paxis == NULL) {
        // reduce all axes
        acc = __riscv_vfmv_v_f_f16m1(0.0f, 1);
        avl = x->ndata;
        for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle16_v_f16m8(px, vl);
            px += vl;
            acc = __riscv_vfredosum_vs_f16m8_f16m1(vx, acc, vl);
        }
        py[0] = __riscv_vfmv_f_s_f16m1_f16(acc);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                acc = __riscv_vfmv_v_f_f16m1(0.0f, 1);
                pxxx = pxx;
                for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vlse16_v_f16m8(pxxx, sizeof(float16_t) * stride, vl);
                    pxxx += vl * stride;
                    acc = __riscv_vfredosum_vs_f16m8_f16m1(vx, acc, vl);
                }
                pxx++;
                *pyy++ = __riscv_vfmv_f_s_f16m1_f16(acc);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceSum_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    float32_t *pxx, *pyy;
    int *paxis = n->priv;
    float32_t res = 0.0f;
    int out_loop_cnt = 1;
    int i, j, k, stride, axis_idx;

    if (paxis == NULL) {
        // reduce all axes
        for (i = 0; i < x->ndata; ++i) {
            res += *px++;
        }
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
        py[0] = res;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                res = 0.0f;
                for (k = 0; k < x->dims[axis_idx]; ++k) {
                    res += *(pxx + k * stride);
                }
                pxx++;
                *pyy++ = res;
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}

void ReduceSum_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas, *pxx, *pxxx;
    float32_t *py = (float32_t *)y->datas, *pyy;
    int *paxis = n->priv;
    int out_loop_cnt = 1;
    int stride, i, j, axis_idx;
    vfloat32m1_t acc;
    vfloat32m8_t vx;
    size_t avl, vl;

    if (paxis == NULL) {
        // reduce all axes
        acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        avl = x->ndata;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle32_v_f32m8(px, vl);
            px += vl;
            acc = __riscv_vfredosum_vs_f32m8_f32m1(vx, acc, vl);
        }
        py[0] = __riscv_vfmv_f_s_f32m1_f32(acc);
        y->ndata = 1;
        y->ndim = 1;
        y->dims[0] = 1;
        y->strides[0] = 1;
    } else {
        // reduce specified axes
        axis_idx = x->ndim - 1 - *paxis;
        stride = x->strides[axis_idx];
        for (i = axis_idx + 1; i < x->ndim; ++i) {
            out_loop_cnt *= x->dims[i];
        }
        // update y data
        for (i = 0; i < out_loop_cnt; ++i) {
            pxx = px + i * stride * x->dims[axis_idx];
            pyy = py + i * stride;
            for (j = 0; j < stride; ++j) {
                avl = x->dims[axis_idx];
                acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                pxxx = pxx;
                for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vlse32_v_f32m8(pxxx, sizeof(float32_t) * stride, vl);
                    pxxx += vl * stride;
                    acc = __riscv_vfredosum_vs_f32m8_f32m1(vx, acc, vl);
                }
                pxx++;
                *pyy++ = __riscv_vfmv_f_s_f32m1_f32(acc);
            }
        }
        // update y dims and strides
        y->ndim = x->ndim - 1;
        for (i = 0; i < axis_idx; ++i) {
            y->dims[i] = x->dims[i];
        }
        for (i = axis_idx; i < x->ndim - 1; ++i) {
            y->dims[i] = x->dims[i + 1];
        }
        for (i = 0; i < y->ndim; ++i) {
            y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
        }
        y->ndata = y->strides[y->ndim - 1] * y->dims[y->ndim - 1];
    }
}