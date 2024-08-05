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
 * https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
 * https://github.com/xboot/libonnx/blob/master/src/default/GRU.c
 */

#include "operators.h"

void Gelu_float16(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  float16_t * px = (float16_t *)x->datas;
  float16_t * py = (float16_t *)y->datas;

  for(size_t i = 0, l = y->ndata; i < l; i++)
    py[i] = 0.5 * px[i] * (1 + tanh(sqrt(2/ PI) * (px[i] + 0.044715 * px[i] * px[i] * px[i])));
}

void Gelu_float32(struct onnx_node_t * n)
{
  struct onnx_tensor_t * x = n->inputs[0];
  struct onnx_tensor_t * y = n->outputs[0];
  float32_t * px = (float32_t *)x->datas;
  float32_t * py = (float32_t *)y->datas;


  for(size_t i = 0, l = y->ndata; i < l; i++)
    py[i] = 0.5 * px[i] * (1 + tanh(sqrt(2/ PI) * (px[i] + 0.044715 * px[i] * px[i] * px[i])));
}
