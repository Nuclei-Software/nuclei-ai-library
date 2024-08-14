/*
 * https://onnx.ai/onnx/operators/onnx__Relu.html#relu
 * https://github.com/shin-mashita/uonnx/blob/main/src/ops/Relu.c
 */

#include "operators.h"

void Relu_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}

void Relu_float16_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t vl;
    vfloat16m8_t vx, vy;
    for (; (vl = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= vl) {
        vx = __riscv_vle16_v_f16m8(px, vl);
        px += vl;
        vx = __riscv_vfmax_vf_f16m8(vx, 0, vl);
        __riscv_vse16_v_f16m8(py, vx, vl);
        py += vl;
    }
}

void Relu_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (px[i] < 0) ? 0 : px[i];
}

void Relu_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t vl;
    vfloat32m8_t vx, vy;
    for (; (vl = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= vl) {
        vx = __riscv_vle32_v_f32m8(px, vl);
        px += vl;
        vx = __riscv_vfmax_vf_f32m8(vx, 0, vl);
        __riscv_vse32_v_f32m8(py, vx, vl);
        py += vl;
    }
}
