/*
 * https://pytorch.org/docs/stable/special.html#torch.special.erf
 * https://github.com/xboot/libonnx/blob/master/src/default/Erf.c
 */

#include "operators.h"

void Erf_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (float16_t)erff((float32_t)px[i]);
}

void Erf_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = erff(px[i]);
}
