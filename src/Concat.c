/**
 * @file Concat.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief references:
 *        https://onnx.ai/onnx/operators/onnx__Concat.html#
 *        https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat
 * @version 0.1
 * @date 2024-07-08
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <string.h>

#include "operators.h"

// TODO(jdqiu): axis is only support 0 or 1, tensor dimension is 2
// assert(x1->ndim == 2 && x2->ndim == 2 &&
// ((x1->dims[0] == x2->dims[0] && axis == 0) ||
//  (x1->dims[1] == x2->dims[1] && axis == 1))
// );

/**
 * NOTE: y->dims is managed by the caller
 *
 * example:
 * x1 = [1, 2]  x2 = [5, 6]
 *      [3, 4]       [7, 8]
 *
 * x1 and x2 concat along axis 0: [1, 2]
 *                                [3, 4]
 *                                [5, 6]
 *                                [7, 8]
 *
 * x1 and x2 concat along axis 1: [1, 2, 5, 6]
 *                                [3, 4, 7, 8]
 *
 */
void Concat_int8(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x;
    struct onnx_tensor_t *y = n->outputs[0];
    int8_t *px, *line_py;
    int8_t *py = (int8_t *)y->datas;
    int axis = *((int *)n->priv);

    if (axis == 0) {
        // concat along axis 0
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            px = (int8_t *)x->datas;
            memcpy(py, px, x->ndata * sizeof(int8_t));
            py += x->ndata;
        }
    } else {
        // concat along axis 1
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            px = (int8_t *)x->datas;
            line_py = py;
            for (int j = 0; j < x->dims[1]; ++j) {
                memcpy(line_py, px, x->dims[0] * sizeof(int8_t));
                line_py += y->dims[0];
                px += x->dims[0];
            }
            py += x->dims[0];
        }
    }
}

void Concat_int8_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x;
    struct onnx_tensor_t *y = n->outputs[0];
    int8_t *px, *pxx, *line_py, *pyy;
    int8_t *py = (int8_t *)y->datas;
    int axis = *((int *)n->priv);
    size_t vl, avl;
    vint8m8_t vx, vy;

    if (axis == 0) {
        // concat along axis 0
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            avl = x->ndata;
            px = (int8_t *)x->datas;
            for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
                vx = __riscv_vle8_v_i8m8(px, vl);
                px += vl;
                __riscv_vse8_v_i8m8(py, vx, vl);
                py += vl;
            }
        }
    } else {
        // concat along axis 1
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            px = (int8_t *)x->datas;
            line_py = py;
            for (int j = 0; j < x->dims[1]; ++j) {
                pxx = px;
                pyy = line_py;
                for (avl = x->dims[0]; avl > 0; avl -= vl) {
                    vl = __riscv_vsetvl_e8m8(avl);
                    vx = __riscv_vle8_v_i8m8(pxx, vl);
                    pxx += vl;
                    __riscv_vse8_v_i8m8(pyy, vx, vl);
                    pyy += vl;
                }
                line_py += y->dims[0];
                px += x->dims[0];
            }
            py += x->dims[0];
        }
    }
}

void Concat_int32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x;
    struct onnx_tensor_t *y = n->outputs[0];
    int32_t *px, *line_py;
    int32_t *py = (int32_t *)y->datas;
    int axis = *((int *)n->priv);

    if (axis == 0) {
        // concat along axis 0
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            px = (int32_t *)x->datas;
            memcpy(py, px, x->ndata * sizeof(int32_t));
            py += x->ndata;
        }
    } else {
        // concat along axis 1
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            px = (int32_t *)x->datas;
            line_py = py;
            for (int j = 0; j < x->dims[1]; ++j) {
                memcpy(line_py, px, x->dims[0] * sizeof(int32_t));
                line_py += y->dims[0];
                px += x->dims[0];
            }
            py += x->dims[0];
        }
    }
}

void Concat_int32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x;
    struct onnx_tensor_t *y = n->outputs[0];
    int32_t *px, *pxx, *line_py, *pyy;
    int32_t *py = (int32_t *)y->datas;
    int axis = *((int *)n->priv);
    size_t vl, avl;
    vint32m8_t vx, vy;

    if (axis == 0) {
        // concat along axis 0
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            avl = x->ndata;
            px = (int32_t *)x->datas;
            for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                vx = __riscv_vle32_v_i32m8(px, vl);
                px += vl;
                __riscv_vse32_v_i32m8(py, vx, vl);
                py += vl;
            }
        }
    } else {
        // concat along axis 1
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            px = (int32_t *)x->datas;
            line_py = py;
            for (int j = 0; j < x->dims[1]; ++j) {
                pxx = px;
                pyy = line_py;
                for (avl = x->dims[0]; avl > 0; avl -= vl) {
                    vl = __riscv_vsetvl_e32m8(avl);
                    vx = __riscv_vle32_v_i32m8(pxx, vl);
                    pxx += vl;
                    __riscv_vse32_v_i32m8(pyy, vx, vl);
                    pyy += vl;
                }
                line_py += y->dims[0];
                px += x->dims[0];
            }
            py += x->dims[0];
        }
    }
}

void Concat_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x;
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px, *line_py;
    float16_t *py = (float16_t *)y->datas;
    int axis = *((int *)n->priv);

    if (axis == 0) {
        // concat along axis 0
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            px = (float16_t *)x->datas;
            memcpy(py, px, x->ndata * sizeof(float16_t));
            py += x->ndata;
        }
    } else {
        // concat along axis 1
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            px = (float16_t *)x->datas;
            line_py = py;
            for (int j = 0; j < x->dims[1]; ++j) {
                memcpy(line_py, px, x->dims[0] * sizeof(float16_t));
                line_py += y->dims[0];
                px += x->dims[0];
            }
            py += x->dims[0];
        }
    }
}

void Concat_float16_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x;
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px, *pxx, *line_py, *pyy;
    float16_t *py = (float16_t *)y->datas;
    int axis = *((int *)n->priv);
    size_t vl, avl;
    vfloat16m8_t vx, vy;

    if (axis == 0) {
        // concat along axis 0
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            avl = x->ndata;
            px = (float16_t *)x->datas;
            for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
                vx = __riscv_vle16_v_f16m8(px, vl);
                px += vl;
                __riscv_vse16_v_f16m8(py, vx, vl);
                py += vl;
            }
        }
    } else {
        // concat along axis 1
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            px = (float16_t *)x->datas;
            line_py = py;
            for (int j = 0; j < x->dims[1]; ++j) {
                pxx = px;
                pyy = line_py;
                for (avl = x->dims[0]; avl > 0; avl -= vl) {
                    vl = __riscv_vsetvl_e16m8(avl);
                    vx = __riscv_vle16_v_f16m8(pxx, vl);
                    pxx += vl;
                    __riscv_vse16_v_f16m8(pyy, vx, vl);
                    pyy += vl;
                }
                line_py += y->dims[0];
                px += x->dims[0];
            }
            py += x->dims[0];
        }
    }
}

void Concat_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x;
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px, *line_py;
    float32_t *py = (float32_t *)y->datas;
    int axis = *((int *)n->priv);

    if (axis == 0) {
        // concat along axis 0
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            px = (float32_t *)x->datas;
            memcpy(py, px, x->ndata * sizeof(float32_t));
            py += x->ndata;
        }
    } else {
        // concat along axis 1
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            px = (float32_t *)x->datas;
            line_py = py;
            for (int j = 0; j < x->dims[1]; ++j) {
                memcpy(line_py, px, x->dims[0] * sizeof(float32_t));
                line_py += y->dims[0];
                px += x->dims[0];
            }
            py += x->dims[0];
        }
    }
}

void Concat_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x;
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px, *pxx, *line_py, *pyy;
    float32_t *py = (float32_t *)y->datas;
    int axis = *((int *)n->priv);
    size_t vl, avl;
    vfloat32m8_t vx, vy;

    if (axis == 0) {
        // concat along axis 0
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            avl = x->ndata;
            px = (float32_t *)x->datas;
            for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                vx = __riscv_vle32_v_f32m8(px, vl);
                px += vl;
                __riscv_vse32_v_f32m8(py, vx, vl);
                py += vl;
            }
        }
    } else {
        // concat along axis 1
        for (int i = 0; i < n->ninput; ++i) {
            x = n->inputs[i];
            px = (float32_t *)x->datas;
            line_py = py;
            for (int j = 0; j < x->dims[1]; ++j) {
                pxx = px;
                pyy = line_py;
                for (avl = x->dims[0]; avl > 0; avl -= vl) {
                    vl = __riscv_vsetvl_e32m8(avl);
                    vx = __riscv_vle32_v_f32m8(pxx, vl);
                    pxx += vl;
                    __riscv_vse32_v_f32m8(pyy, vx, vl);
                    pyy += vl;
                }
                line_py += y->dims[0];
                px += x->dims[0];
            }
            py += x->dims[0];
        }
    }
}
