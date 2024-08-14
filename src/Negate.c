/*
 * https://onnx.ai/onnx/operators/onnx__Neg.html#neg
 * https://github.com/xboot/libonnx/blob/master/src/default/Neg.c
 */

#include "operators.h"

void Negate_int8(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        py[i] = -px[i];
    }
}

void Negate_int8_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t vl;
    vint8m8_t vx;
    for (; (vl = __riscv_vsetvl_e8m8(blkCnt)) > 0; blkCnt -= vl) {
        vx = __riscv_vle8_v_i8m8(px, vl);
        px += vl;
        vx = __riscv_vrsub_vx_i8m8(vx, 0, vl);
        __riscv_vse8_v_i8m8(py, vx, vl);
        py += vl;
    }
}

void Negate_int32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        py[i] = -px[i];
    }
}

void Negate_int32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t vl;
    vint32m8_t vx;
    for (; (vl = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= vl) {
        vx = __riscv_vle32_v_i32m8(px, vl);
        px += vl;
        vx = __riscv_vrsub_vx_i32m8(vx, 0, vl);
        __riscv_vse32_v_i32m8(py, vx, vl);
        py += vl;
    }
}

void Negate_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        py[i] = -px[i];
    }
}

void Negate_float16_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t vl;
    vfloat16m8_t vx;
    for (; (vl = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= vl) {
        vx = __riscv_vle16_v_f16m8(px, vl);
        px += vl;
        vx = __riscv_vfrsub_vf_f16m8(vx, 0.0f, vl);
        __riscv_vse16_v_f16m8(py, vx, vl);
        py += vl;
    }
}

void Negate_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        py[i] = -px[i];
    }
}

void Negate_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t vl;
    vfloat32m8_t vx;
    for (; (vl = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= vl) {
        vx = __riscv_vle32_v_f32m8(px, vl);
        px += vl;
        vx = __riscv_vfrsub_vf_f32m8(vx, 0.0f, vl);
        __riscv_vse32_v_f32m8(py, vx, vl);
        py += vl;
    }
}
