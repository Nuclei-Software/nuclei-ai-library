/**
 * @file Pad.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief references:
 *        https://onnx.ai/onnx/operators/onnx__Pad.html
 *        https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
 * @version 0.1
 * @date 2024-07-10
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <string.h>

#include "operators.h"

typedef enum pad_mode_t {
    PAD_MODE_CONSTANT = 0,
    PAD_MODE_REFLECT = 1,
    PAD_MODE_EDGE = 2
} PadMode;

struct operator_pdata_t {
    PadMode mode;
    struct PadLen {
        int top;
        int bottom;
        int left;
        int right;
    } pads;
    OnnxScalar value;
};

// TODO(jdqiu): Only implement constant pad and dimension of x is 2
// assert(mode == PAD_MODE_CONSTANT && x->ndim == 2)

/**
 * NOTE: y->dims is managed by the caller
 * pad const = 85
 *      pads = [1 2 3 4]
 *         x = [45 49]
 *             [70 41]
 *    result = [ 85 85 85 85 85 85 85 85 85 ]
 *             [ 85 85 85 45 49 85 85 85 85 ]
 *             [ 85 85 85 70 41 85 85 85 85 ]
 *             [ 85 85 85 85 85 85 85 85 85 ]
 *             [ 85 85 85 85 85 85 85 85 85 ]
 *
 */
void Pad_int8(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    size_t len;

    // fill header
    if (pdat->pads.top > 0) {
        len = pdat->pads.top * (pdat->pads.left + pdat->pads.right + x->dims[0]);
        memset(py, pdat->value.v_int8, len);
        py += len;
    }

    // fill body
    for (int i = 0; i < x->dims[1]; ++i) {
        memset(py, pdat->value.v_int8, pdat->pads.left);
        py += pdat->pads.left;
        memcpy(py, px, x->dims[0]);
        py += x->dims[0];
        px += x->dims[0];
        memset(py, pdat->value.v_int8, pdat->pads.right);
        py += pdat->pads.right;
    }

    // fill tail
    if (pdat->pads.bottom > 0) {
        len = pdat->pads.bottom * (pdat->pads.left + pdat->pads.right + x->dims[0]);
        memset(py, pdat->value.v_int8, len);
    }
}

void Pad_int8_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    size_t maxlen, len[2], avl, vl;
    vint8m8_t vx, vx_const;

    len[0] = pdat->pads.top * (pdat->pads.left + pdat->pads.right + x->dims[0]);
    len[1] = pdat->pads.bottom * (pdat->pads.left + pdat->pads.right + x->dims[0]);
    maxlen = len[0] > len[1] ? len[0] : len[1];
    maxlen = maxlen > pdat->pads.left ? maxlen : pdat->pads.left;
    maxlen = maxlen > pdat->pads.right ? maxlen : pdat->pads.right;

    vl = __riscv_vsetvlmax_e8m8();
    if (maxlen >= vl) {
        vx_const = __riscv_vmv_v_x_i8m8(pdat->value.v_int8, vl);
    } else {
        vx_const = __riscv_vmv_v_x_i8m8(pdat->value.v_int8, maxlen);
    }

    // fill header
    for (; (vl = __riscv_vsetvl_e8m8(len[0])) > 0; len[0] -= vl) {
        __riscv_vse8_v_i8m8(py, vx_const, vl);
        py += vl;
    }

    // fill body
    for (int i = 0; i < x->dims[1]; ++i) {
        avl = pdat->pads.left;
        for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
            __riscv_vse8_v_i8m8(py, vx_const, vl);
            py += vl;
        }
        avl = x->dims[0];
        for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle8_v_i8m8(px, vl);
            px += vl;
            __riscv_vse8_v_i8m8(py, vx, vl);
            py += vl;
        }
        avl = pdat->pads.right;
        for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
            __riscv_vse8_v_i8m8(py, vx_const, vl);
            py += vl;
        }
    }

    // fill tail
    for (; (vl = __riscv_vsetvl_e8m8(len[1])) > 0; len[1] -= vl) {
        __riscv_vse8_v_i8m8(py, vx_const, vl);
        py += vl;
    }
}

void Pad_int32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    size_t len;

    // fill header
    if (pdat->pads.top > 0) {
        len = pdat->pads.top * (pdat->pads.left + pdat->pads.right + x->dims[0]);
        for (int i = 0; i < len; i++) {
            py[i] = pdat->value.v_int32;
        }
        py += len;
    }

    // fill body
    for (int i = 0; i < x->dims[1]; ++i) {
        for (int i = 0; i < pdat->pads.left; i++) {
            py[i] = pdat->value.v_int32;
        }
        py += pdat->pads.left;
        memcpy(py, px, x->dims[0] * sizeof(int32_t));
        py += x->dims[0];
        px += x->dims[0];
        for (int i = 0; i < pdat->pads.right; i++) {
            py[i] = pdat->value.v_int32;
        }
        py += pdat->pads.right;
    }

    // fill tail
    if (pdat->pads.bottom > 0) {
        len = pdat->pads.bottom * (pdat->pads.left + pdat->pads.right + x->dims[0]);
        for (int i = 0; i < len; i++) {
            py[i] = pdat->value.v_int32;
        }
    }
}

void Pad_int32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    size_t maxlen, len[2], avl, vl;
    vint32m8_t vx, vx_const;

    len[0] = pdat->pads.top * (pdat->pads.left + pdat->pads.right + x->dims[0]);
    len[1] = pdat->pads.bottom * (pdat->pads.left + pdat->pads.right + x->dims[0]);
    maxlen = len[0] > len[1] ? len[0] : len[1];
    maxlen = maxlen > pdat->pads.left ? maxlen : pdat->pads.left;
    maxlen = maxlen > pdat->pads.right ? maxlen : pdat->pads.right;

    vl = __riscv_vsetvlmax_e32m8();
    if (maxlen >= vl) {
        vx_const = __riscv_vmv_v_x_i32m8(pdat->value.v_int32, vl);
    } else {
        vx_const = __riscv_vmv_v_x_i32m8(pdat->value.v_int32, maxlen);
    }

    // fill header
    for (; (vl = __riscv_vsetvl_e32m8(len[0])) > 0; len[0] -= vl) {
        __riscv_vse32_v_i32m8(py, vx_const, vl);
        py += vl;
    }

    // fill body
    for (int i = 0; i < x->dims[1]; ++i) {
        avl = pdat->pads.left;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            __riscv_vse32_v_i32m8(py, vx_const, vl);
            py += vl;
        }
        avl = x->dims[0];
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle32_v_i32m8(px, vl);
            px += vl;
            __riscv_vse32_v_i32m8(py, vx, vl);
            py += vl;
        }
        avl = pdat->pads.right;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            __riscv_vse32_v_i32m8(py, vx_const, vl);
            py += vl;
        }
    }

    // fill tail
    for (; (vl = __riscv_vsetvl_e32m8(len[1])) > 0; len[1] -= vl) {
        __riscv_vse32_v_i32m8(py, vx_const, vl);
        py += vl;
    }
}

void Pad_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    size_t len;

    // fill header
    if (pdat->pads.top > 0) {
        len = pdat->pads.top * (pdat->pads.left + pdat->pads.right + x->dims[0]);
        for (int i = 0; i < len; i++) {
            py[i] = pdat->value.v_float16;
        }
        py += len;
    }

    // fill body
    for (int i = 0; i < x->dims[1]; ++i) {
        for (int i = 0; i < pdat->pads.left; i++) {
            py[i] = pdat->value.v_float16;
        }
        py += pdat->pads.left;
        memcpy(py, px, x->dims[0] * sizeof(float16_t));
        py += x->dims[0];
        px += x->dims[0];
        for (int i = 0; i < pdat->pads.right; i++) {
            py[i] = pdat->value.v_float16;
        }
        py += pdat->pads.right;
    }

    // fill tail
    if (pdat->pads.bottom > 0) {
        len = pdat->pads.bottom * (pdat->pads.left + pdat->pads.right + x->dims[0]);
        for (int i = 0; i < len; i++) {
            py[i] = pdat->value.v_float16;
        }
    }
}

void Pad_float16_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    size_t maxlen, len[2], avl, vl;
    vfloat16m8_t vx, vx_const;

    len[0] = pdat->pads.top * (pdat->pads.left + pdat->pads.right + x->dims[0]);
    len[1] = pdat->pads.bottom * (pdat->pads.left + pdat->pads.right + x->dims[0]);
    maxlen = len[0] > len[1] ? len[0] : len[1];
    maxlen = maxlen > pdat->pads.left ? maxlen : pdat->pads.left;
    maxlen = maxlen > pdat->pads.right ? maxlen : pdat->pads.right;

    vl = __riscv_vsetvlmax_e16m8();
    if (maxlen >= vl) {
        vx_const = __riscv_vfmv_v_f_f16m8(pdat->value.v_float16, vl);
    } else {
        vx_const = __riscv_vfmv_v_f_f16m8(pdat->value.v_float16, maxlen);
    }

    // fill header
    for (; (vl = __riscv_vsetvl_e16m8(len[0])) > 0; len[0] -= vl) {
        __riscv_vse16_v_f16m8(py, vx_const, vl);
        py += vl;
    }

    // fill body
    for (int i = 0; i < x->dims[1]; ++i) {
        avl = pdat->pads.left;
        for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
            __riscv_vse16_v_f16m8(py, vx_const, vl);
            py += vl;
        }
        avl = x->dims[0];
        for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle16_v_f16m8(px, vl);
            px += vl;
            __riscv_vse16_v_f16m8(py, vx, vl);
            py += vl;
        }
        avl = pdat->pads.right;
        for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
            __riscv_vse16_v_f16m8(py, vx_const, vl);
            py += vl;
        }
    }

    // fill tail
    for (; (vl = __riscv_vsetvl_e16m8(len[1])) > 0; len[1] -= vl) {
        __riscv_vse16_v_f16m8(py, vx_const, vl);
        py += vl;
    }
}

void Pad_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    size_t len;

    // fill header
    if (pdat->pads.top > 0) {
        len = pdat->pads.top * (pdat->pads.left + pdat->pads.right + x->dims[0]);
        for (int i = 0; i < len; i++) {
            py[i] = pdat->value.v_float32;
        }
        py += len;
    }

    // fill body
    for (int i = 0; i < x->dims[1]; ++i) {
        for (int i = 0; i < pdat->pads.left; i++) {
            py[i] = pdat->value.v_float32;
        }
        py += pdat->pads.left;
        memcpy(py, px, x->dims[0] * sizeof(float32_t));
        py += x->dims[0];
        px += x->dims[0];
        for (int i = 0; i < pdat->pads.right; i++) {
            py[i] = pdat->value.v_float32;
        }
        py += pdat->pads.right;
    }

    // fill tail
    if (pdat->pads.bottom > 0) {
        len = pdat->pads.bottom * (pdat->pads.left + pdat->pads.right + x->dims[0]);
        for (int i = 0; i < len; i++) {
            py[i] = pdat->value.v_float32;
        }
    }
}

void Pad_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    size_t maxlen, len[2], avl, vl;
    vfloat32m8_t vx, vx_const;

    len[0] = pdat->pads.top * (pdat->pads.left + pdat->pads.right + x->dims[0]);
    len[1] = pdat->pads.bottom * (pdat->pads.left + pdat->pads.right + x->dims[0]);
    maxlen = len[0] > len[1] ? len[0] : len[1];
    maxlen = maxlen > pdat->pads.left ? maxlen : pdat->pads.left;
    maxlen = maxlen > pdat->pads.right ? maxlen : pdat->pads.right;

    vl = __riscv_vsetvlmax_e32m8();
    if (maxlen >= vl) {
        vx_const = __riscv_vfmv_v_f_f32m8(pdat->value.v_float32, vl);
    } else {
        vx_const = __riscv_vfmv_v_f_f32m8(pdat->value.v_float32, maxlen);
    }

    // fill header
    for (; (vl = __riscv_vsetvl_e32m8(len[0])) > 0; len[0] -= vl) {
        __riscv_vse32_v_f32m8(py, vx_const, vl);
        py += vl;
    }

    // fill body
    for (int i = 0; i < x->dims[1]; ++i) {
        avl = pdat->pads.left;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            __riscv_vse32_v_f32m8(py, vx_const, vl);
            py += vl;
        }
        avl = x->dims[0];
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            vx = __riscv_vle32_v_f32m8(px, vl);
            px += vl;
            __riscv_vse32_v_f32m8(py, vx, vl);
            py += vl;
        }
        avl = pdat->pads.right;
        for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
            __riscv_vse32_v_f32m8(py, vx_const, vl);
            py += vl;
        }
    }

    // fill tail
    for (; (vl = __riscv_vsetvl_e32m8(len[1])) > 0; len[1] -= vl) {
        __riscv_vse32_v_f32m8(py, vx_const, vl);
        py += vl;
    }
}

void *GeneratePadParam(OnnxScalar value, int top, int bottom, int left, int right)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)malloc(sizeof(struct operator_pdata_t));
    pdat->mode = PAD_MODE_CONSTANT;
    pdat->value = value;
    pdat->pads.top = top;
    pdat->pads.bottom = bottom;
    pdat->pads.left = left;
    pdat->pads.right = right;
    return pdat;
}

void FreePadParam(void **pdat)
{
    free(*pdat);
    *pdat = NULL;
}