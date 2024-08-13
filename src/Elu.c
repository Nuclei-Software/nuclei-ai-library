/**
 * @file Softmax.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-07
 *
 * @copyright Copyright (c) 2024
 *
 */

/*
 * https://onnx.ai/onnx/operators/onnx__Elu.html#elu
 * https://github.com/xboot/libonnx/blob/master/src/default/Elu.c
 */

#include "operators.h"
#include "utils.h"

struct operator_pdata_t {
    float32_t alpha;
};

void Elu_float16(struct onnx_node_t * n)
{
   struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  float16_t * px = (float16_t *)x->datas;
  float16_t * py = (float16_t *)y->datas;
  float16_t alpha = (float16_t)pdat->alpha;

  for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? (expf(px[i] ) - 1.) * alpha : px[i];
}

void Elu_float16_rvv(struct onnx_node_t * n)
{
   struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  float16_t * px = (float16_t *)x->datas;
  float16_t * py = (float16_t *)y->datas;
  float16_t alpha = (float16_t)pdat->alpha;

  size_t blkCnt = y->ndata; /* Loop counter */
  size_t vl;
  vfloat16m8_t vx, vy, vz;
  vint16m8_t vx_int;
  vbool2_t mask;
  for (; (vl = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= vl) {
      vx = __riscv_vle16_v_f16m8(px, vl);

      vx = __riscv_vfmul_vf_f16m8(vx, 1.4426950408889634f, vl); // log2(e)
      vx_int = __riscv_vfcvt_rtz_x_f_v_i16m8(vx, vl);
      vx = __riscv_vfsub_vv_f16m8(vx, __riscv_vfcvt_f_x_v_f16m8(vx_int, vl), vl);

      vx_int = __riscv_vadd_vx_i16m8(vx_int, 15, vl);
      vx_int = __riscv_vmul_vx_i16m8(vx_int, (1 << 10), vl);
      vy = __riscv_vreinterpret_v_i16m8_f16m8(vx_int);

      vx = __riscv_vfmul_vf_f16m8(vx, 0.693147180559945f, vl);                        // ln2
      vz = __riscv_vfmul_vf_f16m8(vx, 1.0 / 5040, vl);                                // 1/7!
      vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 720, vl), vl); // 1/6!
      vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 120, vl), vl); // 1/5!
      vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 24, vl), vl);  // 1/4!
      vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 6, vl), vl);   // 1/3!
      vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 2, vl), vl);   // 1/2!
      vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0, vl), vl);       // 1/1!
      vz = __riscv_vfadd_vf_f16m8(vz, 1, vl);

      vy = __riscv_vfmul_vv_f16m8(vy, vz, vl);

      vy = __riscv_vfsub_vf_f16m8(vy, 1.0, vl);
      vy = __riscv_vfmul_vf_f16m8(vy, alpha, vl);

      vx = __riscv_vle16_v_f16m8(px, vl);
      px += vl;
      mask = __riscv_vmflt_vf_f16m8_b2(vx, 0.0, vl);
      vx = __riscv_vmerge_vvm_f16m8(vx, vy, mask, vl);
      __riscv_vse16_v_f16m8(py, vx, vl);
      py += vl;
  }
}

void Elu_float32(struct onnx_node_t * n)
{
   struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  float32_t * px = (float32_t *)x->datas;
  float32_t * py = (float32_t *)y->datas;
  float32_t alpha = pdat->alpha;

  for(size_t i = 0, l = y->ndata; i < l; i++)
		py[i] = (px[i] < 0) ? (expf(px[i]) - 1.) * alpha : px[i];
}

void Elu_float32_rvv(struct onnx_node_t * n)
{
  struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  float32_t * px = (float32_t *)x->datas;
  float32_t * py = (float32_t *)y->datas;
  float32_t alpha = pdat->alpha;

  size_t blkCnt = y->ndata; /* Loop counter */
  size_t vl;
  vfloat32m8_t vx, vy, vz;
  vint32m8_t vx_int;
  vbool4_t mask;
  for (; (vl = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= vl) {
    vx = __riscv_vle32_v_f32m8(px, vl);

    vx = __riscv_vfmul_vf_f32m8(vx, 1.4426950408889634f, vl); // log2(e)
    vx_int = __riscv_vfcvt_rtz_x_f_v_i32m8 (vx, vl);
    vx = __riscv_vfsub_vv_f32m8(vx, __riscv_vfcvt_f_x_v_f32m8(vx_int, vl), vl);

    vx_int = __riscv_vadd_vx_i32m8(vx_int, 127, vl);
    vx_int = __riscv_vmul_vx_i32m8(vx_int, (1 << 23), vl);
    vy = __riscv_vreinterpret_v_i32m8_f32m8 (vx_int);

    vx = __riscv_vfmul_vf_f32m8(vx, 0.693147180559945f, vl);                           // ln2
    vz = __riscv_vfmul_vf_f32m8(vx, 1.0 / 5040, vl);                                   // 1/7!
    vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1.0 / 720, vl), vl);     // 1/6!
    vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1.0 / 120, vl), vl);     // 1/5!
    vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1.0 / 24, vl), vl);      // 1/4!
    vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1.0 / 6, vl), vl);       // 1/3!
    vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1.0 / 2, vl), vl);       // 1/2!
    vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1.0, vl), vl);           // 1/1!
    vz = __riscv_vfadd_vf_f32m8(vz, 1, vl);

    vy = __riscv_vfmul_vv_f32m8(vy, vz, vl);

    vy = __riscv_vfsub_vf_f32m8(vy, 1.0, vl);
    vy = __riscv_vfmul_vf_f32m8(vy, alpha, vl);

    vx = __riscv_vle32_v_f32m8(px, vl);
    px += vl;
    mask = __riscv_vmflt_vf_f32m8_b4(vx, 0.0, vl);
    vx = __riscv_vmerge_vvm_f32m8(vx, vy, mask, vl);
    __riscv_vse32_v_f32m8(py, vx, vl);
    py += vl;
  }
}


void *GenerateEluParam(float32_t alpha)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)MALLOC_ASSERT(sizeof(struct operator_pdata_t));
    pdat->alpha = alpha;
    return pdat;
}

void FreeEluParam(void **pdat)
{
    free(*pdat);
    *pdat = NULL;
}
