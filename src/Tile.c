/**
 * @file Tile.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief references:
 *        https://onnx.ai/onnx/operators/onnx__Tile.html
 *        https://numpy.org/doc/stable/reference/generated/numpy.tile.html#numpy.tile
 * @version 0.1
 * @date 2024-07-11
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <string.h>

#include "operators.h"

/**
 * example0:
 * input:
 *   x = [1, 2]
 *       [3, 4]
 *   T = [1, 2]
 * output:
 *   y = [1, 2, 1, 2]
 *       [3, 4, 3, 4]
 *
 * example1:
 * input:
 *   x = [1, 2]
 *       [3, 4]
 *   T = [2, 1]
 * output:
 *   y = [1, 2]
 *       [3, 4]
 *       [1, 2]
 *       [3, 4]
 *
 * example2:
 * input:
 *   x = [1, 2]
 *       [3, 4]
 *   T = [2, 2]
 * output:
 *   y = [1, 2, 1, 2]
 *       [3, 4, 3, 4]
 *       [1, 2, 1, 2]
 *       [3, 4, 3, 4]
 */
void Tile_int8(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *t = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    int *pt = (int *)t->datas;
    int stride;

    // only support x->ndim == 2
    // assert(x->ndim == t->dims[0] && t->ndim == 1 && y->ndim == x->ndim);

    // update y dims and strides
    for (int i = 0; i < x->ndim; ++i) {
        y->dims[i] = x->dims[i] * pt[x->ndim - 1 - i];
        y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
    }
    y->ndata = y->dims[0] * y->dims[1];

    // update y data
    for (int row = 0; row < x->dims[1]; ++row) {
        for (int i = 0; i < pt[1]; ++i) {
            memcpy(py + i * x->dims[0], px, x->dims[0] * sizeof(int8_t));
        }
        px += x->dims[0];
        py += y->dims[0];
    }

    if (pt[0] > 1) {
        px = y->datas;
        stride = y->strides[1] * x->dims[1];
        for (int i = 0; i < (pt[0] - 1); ++i) {
            memcpy(py + i * stride, px, stride * sizeof(int8_t));
        }
    }
}

void Tile_int8_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *t = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    int8_t *pxx, *pyy;
    int *pt = (int *)t->datas;
    size_t avl, vl;
    vint8m8_t vx;

    // only support x->ndim == 2
    // assert(x->ndim == t->dims[0] && t->ndim == 1 && y->ndim == x->ndim);

    // update y dims and strides
    for (int i = 0; i < x->ndim; ++i) {
        y->dims[i] = x->dims[i] * pt[x->ndim - 1 - i];
        y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
    }
    y->ndata = y->dims[0] * y->dims[1];

    // update y data
    for (int row = 0; row < x->dims[1]; ++row) {
        avl = x->dims[0];
        pxx = px;
        pyy = py;
        for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle8_v_i8m8(pxx, vl);
            for (int i = 0; i < pt[1]; ++i) {
                __riscv_vse8_v_i8m8(pyy + i * x->dims[0], vx, vl);
            }
            pxx += vl;
            pyy += vl;
        }
        px += x->dims[0];
        py += y->dims[0];
    }

    if (pt[0] > 1) {
        px = y->datas;
        avl = y->strides[1] * x->dims[1];
        pxx = px;
        pyy = py;
        for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle8_v_i8m8(pxx, vl);
            for (int i = 0; i < (pt[0] - 1); ++i) {
                __riscv_vse8_v_i8m8(pyy + i * avl, vx, vl);
            }
            pxx += vl;
            pyy += vl;
        }
    }
}

void Tile_int32(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *t = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    int *pt = (int *)t->datas;
    int stride;

    // only support x->ndim == 2
    // assert(x->ndim == t->dims[0] && t->ndim == 1 && y->ndim == x->ndim);

    // update y dims and strides
    for (int i = 0; i < x->ndim; ++i) {
        y->dims[i] = x->dims[i] * pt[x->ndim - 1 - i];
        y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
    }
    y->ndata = y->dims[0] * y->dims[1];

    // update y data
    for (int row = 0; row < x->dims[1]; ++row) {
        for (int i = 0; i < pt[1]; ++i) {
            memcpy(py + i * x->dims[0], px, x->dims[0] * sizeof(int32_t));
        }
        px += x->dims[0];
        py += y->dims[0];
    }

    if (pt[0] > 1) {
        px = y->datas;
        stride = y->strides[1] * x->dims[1];
        for (int i = 0; i < (pt[0] - 1); ++i) {
            memcpy(py + i * stride, px, stride * sizeof(int32_t));
        }
    }
}

void Tile_int32_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *t = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    int32_t *pxx, *pyy;
    int *pt = (int *)t->datas;
    size_t avl, vl;
    vint32m8_t vx;

    // only support x->ndim == 2
    // assert(x->ndim == t->dims[0] && t->ndim == 1 && y->ndim == x->ndim);

    // update y dims and strides
    for (int i = 0; i < x->ndim; ++i) {
        y->dims[i] = x->dims[i] * pt[x->ndim - 1 - i];
        y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
    }
    y->ndata = y->dims[0] * y->dims[1];

    // update y data
    for (int row = 0; row < x->dims[1]; ++row) {
        avl = x->dims[0];
        pxx = px;
        pyy = py;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle32_v_i32m8(pxx, vl);
            for (int i = 0; i < pt[1]; ++i) {
                __riscv_vse32_v_i32m8(pyy + i * x->dims[0], vx, vl);
            }
            pxx += vl;
            pyy += vl;
        }
        px += x->dims[0];
        py += y->dims[0];
    }

    if (pt[0] > 1) {
        px = y->datas;
        avl = y->strides[1] * x->dims[1];
        pxx = px;
        pyy = py;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle32_v_i32m8(pxx, vl);
            for (int i = 0; i < (pt[0] - 1); ++i) {
                __riscv_vse32_v_i32m8(pyy + i * avl, vx, vl);
            }
            pxx += vl;
            pyy += vl;
        }
    }
}

void Tile_float16(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *t = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    int *pt = (int *)t->datas;
    int stride;

    // only support x->ndim == 2
    // assert(x->ndim == t->dims[0] && t->ndim == 1 && y->ndim == x->ndim);

    // update y dims and strides
    for (int i = 0; i < x->ndim; ++i) {
        y->dims[i] = x->dims[i] * pt[x->ndim - 1 - i];
        y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
    }
    y->ndata = y->dims[0] * y->dims[1];

    // update y data
    for (int row = 0; row < x->dims[1]; ++row) {
        for (int i = 0; i < pt[1]; ++i) {
            memcpy(py + i * x->dims[0], px, x->dims[0] * sizeof(float16_t));
        }
        px += x->dims[0];
        py += y->dims[0];
    }

    if (pt[0] > 1) {
        px = y->datas;
        stride = y->strides[1] * x->dims[1];
        for (int i = 0; i < (pt[0] - 1); ++i) {
            memcpy(py + i * stride, px, stride * sizeof(float16_t));
        }
    }
}

void Tile_float16_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *t = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    float16_t *pxx, *pyy;
    int *pt = (int *)t->datas;
    size_t avl, vl;
    vfloat16m8_t vx;

    // only support x->ndim == 2
    // assert(x->ndim == t->dims[0] && t->ndim == 1 && y->ndim == x->ndim);

    // update y dims and strides
    for (int i = 0; i < x->ndim; ++i) {
        y->dims[i] = x->dims[i] * pt[x->ndim - 1 - i];
        y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
    }
    y->ndata = y->dims[0] * y->dims[1];

    // update y data
    for (int row = 0; row < x->dims[1]; ++row) {
        avl = x->dims[0];
        pxx = px;
        pyy = py;
        for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle16_v_f16m8(pxx, vl);
            for (int i = 0; i < pt[1]; ++i) {
                __riscv_vse16_v_f16m8(pyy + i * x->dims[0], vx, vl);
            }
            pxx += vl;
            pyy += vl;
        }
        px += x->dims[0];
        py += y->dims[0];
    }

    if (pt[0] > 1) {
        px = y->datas;
        avl = y->strides[1] * x->dims[1];
        pxx = px;
        pyy = py;
        for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle16_v_f16m8(pxx, vl);
            for (int i = 0; i < (pt[0] - 1); ++i) {
                __riscv_vse16_v_f16m8(pyy + i * avl, vx, vl);
            }
            pxx += vl;
            pyy += vl;
        }
    }
}

void Tile_float32(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *t = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    int *pt = (int *)t->datas;
    int stride;

    // only support x->ndim == 2
    // assert(x->ndim == t->dims[0] && t->ndim == 1 && y->ndim == x->ndim);

    // update y dims and strides
    for (int i = 0; i < x->ndim; ++i) {
        y->dims[i] = x->dims[i] * pt[x->ndim - 1 - i];
        y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
    }
    y->ndata = y->dims[0] * y->dims[1];

    // update y data
    for (int row = 0; row < x->dims[1]; ++row) {
        for (int i = 0; i < pt[1]; ++i) {
            memcpy(py + i * x->dims[0], px, x->dims[0] * sizeof(float32_t));
        }
        px += x->dims[0];
        py += y->dims[0];
    }

    if (pt[0] > 1) {
        px = y->datas;
        stride = y->strides[1] * x->dims[1];
        for (int i = 0; i < (pt[0] - 1); ++i) {
            memcpy(py + i * stride, px, stride * sizeof(float32_t));
        }
    }
}

void Tile_float32_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *t = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    float32_t *pxx, *pyy;
    int *pt = (int *)t->datas;
    size_t avl, vl;
    vfloat32m8_t vx;

    // only support x->ndim == 2
    // assert(x->ndim == t->dims[0] && t->ndim == 1 && y->ndim == x->ndim);

    // update y dims and strides
    for (int i = 0; i < x->ndim; ++i) {
        y->dims[i] = x->dims[i] * pt[x->ndim - 1 - i];
        y->strides[i] = i > 0 ? y->strides[i - 1] * y->dims[i - 1] : 1;
    }
    y->ndata = y->dims[0] * y->dims[1];

    // update y data
    for (int row = 0; row < x->dims[1]; ++row) {
        avl = x->dims[0];
        pxx = px;
        pyy = py;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle32_v_f32m8(pxx, vl);
            for (int i = 0; i < pt[1]; ++i) {
                __riscv_vse32_v_f32m8(pyy + i * x->dims[0], vx, vl);
            }
            pxx += vl;
            pyy += vl;
        }
        px += x->dims[0];
        py += y->dims[0];
    }

    if (pt[0] > 1) {
        px = y->datas;
        avl = y->strides[1] * x->dims[1];
        pxx = px;
        pyy = py;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle32_v_f32m8(pxx, vl);
            for (int i = 0; i < (pt[0] - 1); ++i) {
                __riscv_vse32_v_f32m8(pyy + i * avl, vx, vl);
            }
            pxx += vl;
            pyy += vl;
        }
    }
}