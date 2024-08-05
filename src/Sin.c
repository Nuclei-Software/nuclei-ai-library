/**
 * @file Sin.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

/*
 * https://onnx.ai/onnx/operators/onnx__Sin.html#sin
 * https://github.com/xboot/libonnx/blob/master/src/default/Sin.c
 */

#include "operators.h"

void Sin_float16(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];

  float16_t * px = (float16_t *)x->datas;
  float16_t * py = (float16_t *)y->datas;

  for(size_t i = 0, l = y->ndata; i < l; i++)
  {
    py[i] = (float16_t)sinf((float32_t)px[i]);
  }
}

void Sin_float16_rvv(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];

  float16_t * px = (float16_t *)x->datas;
  float16_t * py = (float16_t *)y->datas;

  size_t vblkCnt = y->ndata;                               /* Loop counter */
  size_t vl;
  vfloat32m8_t vx, vy, vz;
  vint32m8_t vx_int;
  vbool4_t mask;
  // x - 1/6 x^3 + 1/120 x^5 - 1/5040 x^7 + 1/9! x^9...
  for (; (vl = __riscv_vsetvl_e32m8(vblkCnt)) > 0; vblkCnt -= vl) {
    // Note: Because of the accuracy of float16, should use float32
    vx = __riscv_vfwadd_vf_f32m8(__riscv_vle16_v_f16m4(px, vl), 0.0, vl);
    px += vl;
    vx = __riscv_vfmul_vf_f32m8(vx, 1.0/(2 * PI), vl);
    vx_int = __riscv_vfcvt_rtz_x_f_v_i32m8 (vx, vl);
    vx = __riscv_vfsub_vv_f32m8(vx, __riscv_vfcvt_f_x_v_f32m8(vx_int, vl), vl);
    vx = __riscv_vfmul_vf_f32m8(vx, 2 * PI, vl);

    mask = __riscv_vmfgt_vf_f32m8_b4(vx, PI, vl);
    vx = __riscv_vfadd_vf_f32m8_mu (mask, vx, vx, -2 * PI, vl);

    mask = __riscv_vmflt_vf_f32m8_b4(vx, -PI, vl);
    vx = __riscv_vfadd_vf_f32m8_mu (mask, vx, vx, 2 * PI, vl);

    vy = __riscv_vfmul_vv_f32m8(vx, vx, vl);
    vz = __riscv_vfmul_vf_f32m8(vy, 1.0 / 362880, vl);                                // 1/9!
    vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, -1.0 / 5040, vl), vl); // 1/7!
    vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, 1.0 / 120, vl), vl);   // 1/5!
    vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, -1.0 / 6, vl), vl);    // 1/3!
    vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1, vl), vl);

    __riscv_vse16_v_f16m4(py, __riscv_vfncvt_f_f_w_f16m4(vz, vl), vl);
    py += vl;
  }
}

void Sin_float32(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];

  float32_t * px = (float32_t *)x->datas;
  float32_t * py = (float32_t *)y->datas;

  for(size_t i = 0, l = y->ndata; i < l; i++)
  {
    py[i] = sinf(px[i]);
  }
}

void Sin_float32_rvv(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];

  float32_t * px = (float32_t *)x->datas;
  float32_t * py = (float32_t *)y->datas;

  size_t vblkCnt = y->ndata;                               /* Loop counter */
  size_t vl;
  vfloat32m8_t vx, vy, vz;
  vint32m8_t vx_int;
  vbool4_t mask;
  // x - 1/6 x^3 + 1/120 x^5 - 1/5040 x^7 + 1/9! x^9...
  for (; (vl = __riscv_vsetvl_e32m8(vblkCnt)) > 0; vblkCnt -= vl) {
    vx = __riscv_vle32_v_f32m8(px, vl);
    px += vl;
    vx = __riscv_vfmul_vf_f32m8(vx, 1.0/(2 * PI), vl);
    vx_int = __riscv_vfcvt_rtz_x_f_v_i32m8 (vx, vl);
    vx = __riscv_vfsub_vv_f32m8(vx, __riscv_vfcvt_f_x_v_f32m8(vx_int, vl), vl);
    vx = __riscv_vfmul_vf_f32m8(vx, 2 * PI, vl);

    mask = __riscv_vmfgt_vf_f32m8_b4(vx, PI, vl);
    vx = __riscv_vfadd_vf_f32m8_mu (mask, vx, vx, -2 * PI, vl);

    mask = __riscv_vmflt_vf_f32m8_b4(vx, -PI, vl);
    vx = __riscv_vfadd_vf_f32m8_mu (mask, vx, vx, 2 * PI, vl);

    vy = __riscv_vfmul_vv_f32m8(vx, vx, vl);
    vz = __riscv_vfmul_vf_f32m8(vy, 1.0 / 362880, vl);                               // 1/9!
    vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, -1.0 / 5040, vl), vl); // 1/7!
    vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, 1.0 / 120, vl), vl);   // 1/5!
    vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, -1.0 / 6, vl), vl);    // 1/3!
    vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1, vl), vl);

    __riscv_vse32_v_f32m8(py, vz, vl);
    py += vl;
  }
}
