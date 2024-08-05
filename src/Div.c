/**
 * @file Div.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

/*
 * https://onnx.ai/onnx/operators/onnx__Div.html#div
 * https://github.com/xboot/libonnx/blob/master/src/default/Div.c
 */

#include "operators.h"

void Div_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    float16_t *py = (float16_t *)y->datas;
    float16_t *pa = (float16_t *)a->datas;
    float16_t *pb = (float16_t *)b->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        py[i] = pa[i] / pb[i];
    }
}

void Div_float16_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    float16_t *py = (float16_t *)y->datas;
    float16_t *pa = (float16_t *)a->datas;
    float16_t *pb = (float16_t *)b->datas;

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t l;
    vfloat16m8_t vx, vy;
    for (; (l = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= l) {
        vx = __riscv_vle16_v_f16m8(pa, l);
        vy = __riscv_vle16_v_f16m8(pb, l);
        pa += l;
        pb += l;
        __riscv_vse16_v_f16m8(py, __riscv_vfdiv_vv_f16m8(vx, vy, l), l);
        py += l;
    }
}

void Div_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    float32_t *py = (float32_t *)y->datas;
    float32_t *pa = (float32_t *)a->datas;
    float32_t *pb = (float32_t *)b->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        py[i] = pa[i] / pb[i];
    }
}

void Div_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    float32_t *py = (float32_t *)y->datas;
    float32_t *pa = (float32_t *)a->datas;
    float32_t *pb = (float32_t *)b->datas;

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t l;
    vfloat32m8_t vx, vy;

    for (; (l = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= l) {
        vx = __riscv_vle32_v_f32m8(pa, l);
        vy = __riscv_vle32_v_f32m8(pb, l);
        pa += l;
        pb += l;
        __riscv_vse32_v_f32m8(py, __riscv_vfdiv_vv_f32m8(vx, vy, l), l);
        py += l;
    }
}
