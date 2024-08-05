/**
 * @file Log.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

/*
 * https://onnx.ai/onnx/operators/onnx__Log.html#log
 * https://github.com/xboot/libonnx/blob/master/src/default/Log.c
 */

#include "operators.h"

void Log_float16(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];

  float16_t * px = (float16_t *)x->datas;
  float16_t * py = (float16_t *)y->datas;

  for(size_t i = 0, l = y->ndata; i < l; i++)
  {
    py[i] = (float16_t)logf((float32_t)px[i]);
  }
}

void Log_float16_rvv(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];

  float16_t * px = (float16_t *)x->datas;
  float16_t * py = (float16_t *)y->datas;

  size_t vblkCnt = y->ndata;                               /* Loop counter */
  size_t vl;
  vfloat16m8_t vx, vy, vz, v1;
  vint16m8_t vx_int, vk;

  for (; (vl = __riscv_vsetvl_e16m8(vblkCnt)) > 0; vblkCnt -= vl) {
    vx = __riscv_vle16_v_f16m8(px, vl);
    px += vl;
    vx_int = __riscv_vreinterpret_v_f16m8_i16m8 (vx);
    vk = __riscv_vsra_vx_i16m8(__riscv_vsub_vx_i16m8(vx_int, (15 << 10), vl), 10, vl);

    vz = __riscv_vfadd_vf_f16m8(__riscv_vfcvt_f_x_v_f16m8(vk, vl), 0.5f, vl);
    vz = __riscv_vfmul_vf_f16m8(vz, 0.6931471805599453f, vl); // ln2

    vx_int = __riscv_vor_vx_i16m8(__riscv_vand_vx_i16m8(vx_int, 0x3ff, vl), (15  << 10), vl);
    vy = __riscv_vreinterpret_v_i16m8_f16m8(vx_int);

    vy = __riscv_vfdiv_vv_f16m8 (__riscv_vfsub_vf_f16m8(vy, 1.41421356237309f,vl), __riscv_vfadd_vf_f16m8(vy, 1.41421356237309f,vl), vl);

    vx = __riscv_vfmul_vv_f16m8(vy, vy, vl);
    v1 = __riscv_vfmul_vf_f16m8(vx, 1.0 / 7, vl);
    v1 = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(v1, 1.0 / 5, vl), vl);
    v1 = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(v1, 1.0 / 3, vl), vl);
    v1 = __riscv_vfmul_vv_f16m8(vy, __riscv_vfadd_vf_f16m8(v1, 1.0, vl), vl);
    vx = __riscv_vfmul_vf_f16m8(v1, 2.0, vl);

    vy = __riscv_vfadd_vv_f16m8(vx, vz, vl);
    __riscv_vse16_v_f16m8(py, vy, vl);
    py += vl;
  }
}

void Log_float32(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];

  float32_t * px = (float32_t *)x->datas;
  float32_t * py = (float32_t *)y->datas;

  for(size_t i = 0, l = y->ndata; i < l; i++)
  {
    py[i] = logf(px[i]);
  }
}

void Log_float32_rvv(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];

  float32_t * px = (float32_t *)x->datas;
  float32_t * py = (float32_t *)y->datas;

  size_t vblkCnt = y->ndata;                               /* Loop counter */
  size_t vl;
  vfloat32m8_t vx, vy, vz, v1;
  vint32m8_t vx_int, vk;

  for (; (vl = __riscv_vsetvl_e32m8(vblkCnt)) > 0; vblkCnt -= vl) {
    vx = __riscv_vle32_v_f32m8(px, vl);
    px += vl;
    vx_int = __riscv_vreinterpret_v_f32m8_i32m8 (vx);
    vk = __riscv_vsra_vx_i32m8(__riscv_vsub_vx_i32m8(vx_int, (127 << 23), vl), 23, vl);

    vz = __riscv_vfadd_vf_f32m8(__riscv_vfcvt_f_x_v_f32m8(vk, vl), 0.5f, vl);
    vz = __riscv_vfmul_vf_f32m8(vz, 0.6931471805599453f, vl); // ln2

    vx_int = __riscv_vor_vx_i32m8(__riscv_vand_vx_i32m8(vx_int, 0x7fffff, vl), (127 << 23), vl);
    vy = __riscv_vreinterpret_v_i32m8_f32m8(vx_int);

    vy = __riscv_vfdiv_vv_f32m8 (__riscv_vfsub_vf_f32m8(vy, 1.41421356237309f,vl), __riscv_vfadd_vf_f32m8(vy, 1.41421356237309f,vl), vl);

    vx = __riscv_vfmul_vv_f32m8(vy, vy, vl);
    v1 = __riscv_vfmul_vf_f32m8(vx, 1.0 / 7, vl);
    v1 = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(v1, 1.0 / 5, vl), vl);
    v1 = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(v1, 1.0 / 3, vl), vl);
    v1 = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(v1, 1.0, vl), vl);
    vx = __riscv_vfmul_vf_f32m8(v1, 2.0, vl);

    vy = __riscv_vfadd_vv_f32m8(vx, vz, vl);
    __riscv_vse32_v_f32m8(py, vy, vl);
    py += vl;
  }
}
