/**
 * @file Clamp.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-07
 *
 * @copyright Copyright (c) 2024
 *
 */

/*
 * https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.functional.clamp.html#clamp
 */

#include "operators.h"
#include "utils.h"

struct operator_pdata_t {
	union onnx_scalar_t min;
	union onnx_scalar_t max;
};

void Clamp_int8(struct onnx_node_t * n)
{
  struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  int8_t * px = (int8_t *)x->datas;
  int8_t * py = (int8_t *)y->datas;
  int8_t max = (int8_t)pdat->max.v_int8;
  int8_t min = (int8_t)pdat->min.v_int8;

  for(size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < min) {
      py[i] = min;
    } else if (px[i] > max) {
      py[i] = max;
    } else {
      py[i] = px[i];
    }
  }
}

void Clamp_int8_rvv(struct onnx_node_t * n)
{
  struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  int8_t * px = (int8_t *)x->datas;
  int8_t * py = (int8_t *)y->datas;
  int8_t max = (int8_t)pdat->max.v_int8;
  int8_t min = (int8_t)pdat->min.v_int8;

  size_t blkCnt = y->ndata; /* Loop counter */
  size_t vl;
  vint8m8_t vx, vy;
  for (; (vl = __riscv_vsetvl_e8m8(blkCnt)) > 0; blkCnt -= vl) {
      vx = __riscv_vle8_v_i8m8(px, vl);
      px += vl;
      vx = __riscv_vmax_vx_i8m8(vx, min, vl);
      vx = __riscv_vmin_vx_i8m8(vx, max, vl);
      __riscv_vse8_v_i8m8(py, vx, vl);
      py += vl;
  }
}

void Clamp_int32(struct onnx_node_t * n)
{
  struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  int32_t * px = (int32_t *)x->datas;
  int32_t * py = (int32_t *)y->datas;
  int32_t max = (int32_t)pdat->max.v_int32;
  int32_t min = (int32_t)pdat->min.v_int32;

  for(size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < min) {
      py[i] = min;
    } else if (px[i] > max) {
      py[i] = max;
    } else {
      py[i] = px[i];
    }
  }
}

void Clamp_int32_rvv(struct onnx_node_t * n)
{
  struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  int32_t * px = (int32_t *)x->datas;
  int32_t * py = (int32_t *)y->datas;
  int32_t max = (int32_t)pdat->max.v_int32;
  int32_t min = (int32_t)pdat->min.v_int32;

  size_t blkCnt = y->ndata; /* Loop counter */
  size_t vl;
  vint32m8_t vx, vy;
  for (; (vl = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= vl) {
      vx = __riscv_vle32_v_i32m8(px, vl);
      px += vl;
      vx = __riscv_vmax_vx_i32m8(vx, min, vl);
      vx = __riscv_vmin_vx_i32m8(vx, max, vl);
      __riscv_vse32_v_i32m8(py, vx, vl);
      py += vl;
  }
}

void Clamp_float16(struct onnx_node_t * n)
{
  struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  float16_t * px = (float16_t *)x->datas;
  float16_t * py = (float16_t *)y->datas;
  float16_t max = (float16_t)pdat->max.v_float16;
  float16_t min = (float16_t)pdat->min.v_float16;

  for(size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < min) {
      py[i] = min;
    } else if (px[i] > max) {
      py[i] = max;
    } else {
      py[i] = px[i];
    }
  }
}

void Clamp_float16_rvv(struct onnx_node_t * n)
{
  struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  float16_t * px = (float16_t *)x->datas;
  float16_t * py = (float16_t *)y->datas;
  float16_t max = (float16_t)pdat->max.v_float16;
  float16_t min = (float16_t)pdat->min.v_float16;

  size_t blkCnt = y->ndata; /* Loop counter */
  size_t vl;
  vfloat16m8_t vx, vy;
  for (; (vl = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= vl) {
      vx = __riscv_vle16_v_f16m8(px, vl);
      px += vl;
      vx = __riscv_vfmax_vf_f16m8(vx, min, vl);
      vx = __riscv_vfmin_vf_f16m8(vx, max, vl);
      __riscv_vse16_v_f16m8(py, vx, vl);
      py += vl;
  }
}

void Clamp_float32(struct onnx_node_t * n)
{
  struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  float32_t * px = (float32_t *)x->datas;
  float32_t * py = (float32_t *)y->datas;
  float32_t max = (float32_t)pdat->max.v_float32;
  float32_t min = (float32_t)pdat->min.v_float32;

  for(size_t i = 0, l = y->ndata; i < l; i++) {
    if (px[i] < min) {
      py[i] = min;
    } else if (px[i] > max) {
      py[i] = max;
    } else {
      py[i] = px[i];
    }
  }
}

void Clamp_float32_rvv(struct onnx_node_t * n)
{
  struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  float32_t * px = (float32_t *)x->datas;
  float32_t * py = (float32_t *)y->datas;
  float32_t max = (float32_t)pdat->max.v_float32;
  float32_t min = (float32_t)pdat->min.v_float32;

  size_t blkCnt = y->ndata; /* Loop counter */
  size_t vl;
  vfloat32m8_t vx, vy;
  for (; (vl = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= vl) {
      vx = __riscv_vle32_v_f32m8(px, vl);
      px += vl;
      vx = __riscv_vfmax_vf_f32m8(vx, min, vl);
      vx = __riscv_vfmin_vf_f32m8(vx, max, vl);
      __riscv_vse32_v_f32m8(py, vx, vl);
      py += vl;
  }
}

void *GenerateClampParam(OnnxScalar min, OnnxScalar max)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)MALLOC_ASSERT(sizeof(struct operator_pdata_t));
    pdat->min = min;
    pdat->max = max;
    return pdat;
}

void FreeClampParam(void **pdat)
{
    free(*pdat);
    *pdat = NULL;
}
