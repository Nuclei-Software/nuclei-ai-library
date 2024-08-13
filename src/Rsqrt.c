/**
 * @file Rsqrt.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

/*
 * https://pytorch.org/docs/stable/generated/torch.rsqrt.html
 */

#include "operators.h"

void Rsqrt_float16(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];

  float16_t * px = (float16_t *)x->datas;
  float16_t * py = (float16_t *)y->datas;

  for(size_t i = 0, l = y->ndata; i < l; i++)
  {
    py[i] = (float16_t)1.0 / (float16_t)sqrtf((float32_t)px[i]);
  }
}

void Rsqrt_float16_rvv(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];

  float16_t * px = (float16_t *)x->datas;
  float16_t * py = (float16_t *)y->datas;

  size_t vblkCnt = y->ndata;                               /* Loop counter */
  size_t vl;
  vfloat16m8_t vx, vy;
  for (; (vl = __riscv_vsetvl_e16m8(vblkCnt)) > 0; vblkCnt -= vl) {
    vx = __riscv_vle16_v_f16m8(px, vl);
    px += vl;
    vy =  __riscv_vfsqrt_v_f16m8(vx, vl);
    // vy = __riscv_vfrec7_v_f16m8(vy, vl);
    vy = __riscv_vfrdiv_vf_f16m8(vy, 1, vl);
    __riscv_vse16_v_f16m8(py, vy, vl);
    py += vl;
  }
}

void Rsqrt_float32(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];

  float32_t * px = (float32_t *)x->datas;
  float32_t * py = (float32_t *)y->datas;

  for(size_t i = 0, l = y->ndata; i < l; i++)
  {
    py[i] = 1.0 / sqrtf(px[i]);
  }
}

void Rsqrt_float32_rvv(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];

  float32_t * px = (float32_t *)x->datas;
  float32_t * py = (float32_t *)y->datas;

  size_t vblkCnt = y->ndata;                               /* Loop counter */
  size_t vl;
  vfloat32m8_t vx, vy;
  for (; (vl = __riscv_vsetvl_e32m8(vblkCnt)) > 0; vblkCnt -= vl) {
    vx = __riscv_vle32_v_f32m8(px, vl);
    px += vl;
    vy =  __riscv_vfsqrt_v_f32m8(vx, vl);
    // vy = __riscv_vfrec7_v_f32m8(vy, vl);
    vy = __riscv_vfrdiv_vf_f32m8(vy, 1, vl);
    __riscv_vse32_v_f32m8(py, vy, vl);
    py += vl;
  }
}
