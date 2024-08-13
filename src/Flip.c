/**
 * @file Flip.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief references:
 *        https://pytorch.org/docs/stable/generated/torch.flip.html
 * @version 0.1
 * @date 2024-07-10
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <string.h>

#include "operators.h"
#include "utils.h"

struct operator_pdata_t {
    int axis[2]; /*! axis[0] != 0 -> flip along axis 0, axis[1] != 0 -> flip along axis 1*/
};

// TODO(jdqiu): axis is only support 0 or 1 or both
// assert(x1->ndim == 2 && x2->ndim == 2);

/**
 * example:
 * x = [1, 2]
 *     [3, 4]
 *
 * x flip along axis 0: [3, 4]
 *                      [1, 2]
 *
 * x flip along axis 1: [2, 1]
 *                      [4, 3]
 *
 * x flip along both axis: [4, 3]
 *                         [2, 1]
 *
 */

void Flip_int8(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    int8_t *pxx, *pyy;
    int *p = ((struct operator_pdata_t *)node->priv)->axis;
    int cols = x->dims[0];
    int rows = x->dims[1];

    if (p[0] != 0 && p[1] != 0) {
        // flip both
        py = py + y->ndata - 1;
        for (int i = 0; i < x->ndata; ++i) {
            *py-- = *px++;
        }
    } else {
        if (p[0] != 0) {
            // flip along axis 0
            py += cols * (rows - 1);
            for (int i = 0; i < rows; ++i) {
                memcpy(py, px, cols * sizeof(int8_t));
                px += cols;
                py -= cols;
            }
        }
        if (p[1] != 0) {
            // flip along axis 1
            for (int i = 0; i < rows; ++i) {
                pxx = px + i * cols;
                pyy = py + i * cols + cols - 1;
                for (int j = 0; j < cols; ++j) {
                    *pyy-- = *pxx++;
                }
            }
        }
    }
}

void Flip_int8_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    int8_t *pxx, *pyy;
    int *p = ((struct operator_pdata_t *)node->priv)->axis;
    int cols = x->dims[0];
    int rows = x->dims[1];
    size_t avl, vl;
    vint8m8_t vx;

    if (p[0] != 0 && p[1] != 0) {
        // flip both
        avl = node->inputs[0]->ndata;
        py = py + y->ndata - 1;
        for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle8_v_i8m8(px, vl);
            px += vl;
            __riscv_vsse8_v_i8m8(py, -1 * sizeof(int8_t), vx, vl);
            py -= vl;
        }
    } else {
        if (p[0] != 0) {
            // flip along axis 0
            py += cols * (rows - 1);
            for (int i = 0; i < rows; ++i) {
                avl = cols;
                pxx = px;
                pyy = py;
                for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vle8_v_i8m8(pxx, vl);
                    pxx += vl;
                    __riscv_vse8_v_i8m8(pyy, vx, vl);
                    pyy += vl;
                }
                px += cols;
                py -= cols;
            }
        }
        if (p[1] != 0) {
            // flip along axis 1
            for (int i = 0; i < rows; ++i) {
                avl = cols;
                pxx = px + i * cols;
                pyy = py + i * cols + cols - 1;
                for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vle8_v_i8m8(pxx, vl);
                    pxx += vl;
                    __riscv_vsse8_v_i8m8(pyy, -1 * sizeof(int8_t), vx, vl);
                    pyy -= vl;
                }
            }
        }
    }
}

void Flip_int32(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    int32_t *pxx, *pyy;
    int *p = ((struct operator_pdata_t *)node->priv)->axis;
    int cols = x->dims[0];
    int rows = x->dims[1];

    if (p[0] != 0 && p[1] != 0) {
        // flip both
        py = py + y->ndata - 1;
        for (int i = 0; i < x->ndata; ++i) {
            *py-- = *px++;
        }
    } else {
        if (p[0] != 0) {
            // flip along axis 0
            py += cols * (rows - 1);
            for (int i = 0; i < rows; ++i) {
                memcpy(py, px, cols * sizeof(int32_t));
                px += cols;
                py -= cols;
            }
        }
        if (p[1] != 0) {
            // flip along axis 1
            for (int i = 0; i < rows; ++i) {
                pxx = px + i * cols;
                pyy = py + i * cols + cols - 1;
                for (int j = 0; j < cols; ++j) {
                    *pyy-- = *pxx++;
                }
            }
        }
    }
}

void Flip_int32_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    int32_t *pxx, *pyy;
    int *p = ((struct operator_pdata_t *)node->priv)->axis;
    int cols = x->dims[0];
    int rows = x->dims[1];
    size_t avl, vl;
    vint32m8_t vx;

    if (p[0] != 0 && p[1] != 0) {
        // flip both
        avl = node->inputs[0]->ndata;
        py = py + y->ndata - 1;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle32_v_i32m8(px, vl);
            px += vl;
            __riscv_vsse32_v_i32m8(py, -1 * sizeof(int32_t), vx, vl);
            py -= vl;
        }
    } else {
        if (p[0] != 0) {
            // flip along axis 0
            py += cols * (rows - 1);
            for (int i = 0; i < rows; ++i) {
                avl = cols;
                pxx = px;
                pyy = py;
                for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vle32_v_i32m8(pxx, vl);
                    pxx += vl;
                    __riscv_vse32_v_i32m8(pyy, vx, vl);
                    pyy += vl;
                }
                px += cols;
                py -= cols;
            }
        }
        if (p[1] != 0) {
            // flip along axis 1
            for (int i = 0; i < rows; ++i) {
                avl = cols;
                pxx = px + i * cols;
                pyy = py + i * cols + cols - 1;
                for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vle32_v_i32m8(pxx, vl);
                    pxx += vl;
                    __riscv_vsse32_v_i32m8(pyy, -1 * sizeof(int32_t), vx, vl);
                    pyy -= vl;
                }
            }
        }
    }
}

void Flip_float16(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    float16_t *pxx, *pyy;
    int *p = ((struct operator_pdata_t *)node->priv)->axis;
    int cols = x->dims[0];
    int rows = x->dims[1];

    if (p[0] != 0 && p[1] != 0) {
        // flip both
        py = py + y->ndata - 1;
        for (int i = 0; i < x->ndata; ++i) {
            *py-- = *px++;
        }
    } else {
        if (p[0] != 0) {
            // flip along axis 0
            py += cols * (rows - 1);
            for (int i = 0; i < rows; ++i) {
                memcpy(py, px, cols * sizeof(float16_t));
                px += cols;
                py -= cols;
            }
        }
        if (p[1] != 0) {
            // flip along axis 1
            for (int i = 0; i < rows; ++i) {
                pxx = px + i * cols;
                pyy = py + i * cols + cols - 1;
                for (int j = 0; j < cols; ++j) {
                    *pyy-- = *pxx++;
                }
            }
        }
    }
}

void Flip_float16_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    float16_t *pxx, *pyy;
    int *p = ((struct operator_pdata_t *)node->priv)->axis;
    int cols = x->dims[0];
    int rows = x->dims[1];
    size_t avl, vl;
    vfloat16m8_t vx;

    if (p[0] != 0 && p[1] != 0) {
        // flip both
        avl = node->inputs[0]->ndata;
        py = py + y->ndata - 1;
        for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle16_v_f16m8(px, vl);
            px += vl;
            __riscv_vsse16_v_f16m8(py, -1 * sizeof(float16_t), vx, vl);
            py -= vl;
        }
    } else {
        if (p[0] != 0) {
            // flip along axis 0
            py += cols * (rows - 1);
            for (int i = 0; i < rows; ++i) {
                avl = cols;
                pxx = px;
                pyy = py;
                for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vle16_v_f16m8(pxx, vl);
                    pxx += vl;
                    __riscv_vse16_v_f16m8(pyy, vx, vl);
                    pyy += vl;
                }
                px += cols;
                py -= cols;
            }
        }
        if (p[1] != 0) {
            // flip along axis 1
            for (int i = 0; i < rows; ++i) {
                avl = cols;
                pxx = px + i * cols;
                pyy = py + i * cols + cols - 1;
                for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vle16_v_f16m8(pxx, vl);
                    pxx += vl;
                    __riscv_vsse16_v_f16m8(pyy, -1 * sizeof(float16_t), vx, vl);
                    pyy -= vl;
                }
            }
        }
    }
}

void Flip_float32(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    float32_t *pxx, *pyy;
    int *p = ((struct operator_pdata_t *)node->priv)->axis;
    int cols = x->dims[0];
    int rows = x->dims[1];

    if (p[0] != 0 && p[1] != 0) {
        // flip both
        py = py + y->ndata - 1;
        for (int i = 0; i < x->ndata; ++i) {
            *py-- = *px++;
        }
    } else {
        if (p[0] != 0) {
            // flip along axis 0
            py += cols * (rows - 1);
            for (int i = 0; i < rows; ++i) {
                memcpy(py, px, cols * sizeof(float32_t));
                px += cols;
                py -= cols;
            }
        }
        if (p[1] != 0) {
            // flip along axis 1
            for (int i = 0; i < rows; ++i) {
                pxx = px + i * cols;
                pyy = py + i * cols + cols - 1;
                for (int j = 0; j < cols; ++j) {
                    *pyy-- = *pxx++;
                }
            }
        }
    }
}

void Flip_float32_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *y = node->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    float32_t *pxx, *pyy;
    int *p = ((struct operator_pdata_t *)node->priv)->axis;
    int cols = x->dims[0];
    int rows = x->dims[1];
    size_t avl, vl;
    vfloat32m8_t vx;

    if (p[0] != 0 && p[1] != 0) {
        // flip both
        avl = node->inputs[0]->ndata;
        py = py + y->ndata - 1;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle32_v_f32m8(px, vl);
            px += vl;
            __riscv_vsse32_v_f32m8(py, -1 * sizeof(float32_t), vx, vl);
            py -= vl;
        }
    } else {
        if (p[0] != 0) {
            // flip along axis 0
            py += cols * (rows - 1);
            for (int i = 0; i < rows; ++i) {
                avl = cols;
                pxx = px;
                pyy = py;
                for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vle32_v_f32m8(pxx, vl);
                    pxx += vl;
                    __riscv_vse32_v_f32m8(pyy, vx, vl);
                    pyy += vl;
                }
                px += cols;
                py -= cols;
            }
        }
        if (p[1] != 0) {
            // flip along axis 1
            for (int i = 0; i < rows; ++i) {
                avl = cols;
                pxx = px + i * cols;
                pyy = py + i * cols + cols - 1;
                for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                    vx = __riscv_vle32_v_f32m8(pxx, vl);
                    pxx += vl;
                    __riscv_vsse32_v_f32m8(pyy, -1 * sizeof(float32_t), vx, vl);
                    pyy -= vl;
                }
            }
        }
    }
}

void *GenerateFlipParam(int flip_axis0, int flip_axis1)
{
    struct operator_pdata_t *param = (struct operator_pdata_t *)MALLOC_ASSERT(sizeof(struct operator_pdata_t));
    param->axis[0] = flip_axis0;
    param->axis[1] = flip_axis1;
    return param;
}

void FreeFlipParam(void **pdat)
{
    free(*pdat);
    *pdat = NULL;
}