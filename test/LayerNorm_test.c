/**
 * @file LayerNorm_test.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "utils.h"

#define N 1
#define D 4096

BENCH_DECLARE_VAR()
int test_layernormalization_f32(void)
{
    struct onnx_node_t *node;
    float32_t golden[N * D];
    float32_t opt[N * D];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_CHECKED(sizeof(struct onnx_node_t));
    node->priv = GenerateLayerNormParam(1e-05f, 0.9f);

    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = N;
    node->inputs[0]->dims[1] = D;
    node->inputs[0]->ndata = node->inputs[0]->dims[0] * node->inputs[0]->dims[1];
    node->inputs[0]->datas = MALLOC_CHECKED(sizeof(float32_t) * node->inputs[0]->ndata);

    float32_t *p = (float32_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = N * D;
    node->outputs[0]->datas = MALLOC_CHECKED(sizeof(float32_t) * node->outputs[0]->ndata);

    BENCH_START(LayerNormalization_float32);
    LayerNormalization_float32(node);
    BENCH_END(LayerNormalization_float32);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(LayerNormalization_float32_rvv);
    LayerNormalization_float32_rvv(node);
    BENCH_END(LayerNormalization_float32_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    ret |= verify_results_f32(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]);
    free(node->outputs);
    FreeLayerNormParam(&node->priv);
    free(node);

    return ret;
}

int test_layernormalization_f16(void)
{
    struct onnx_node_t *node;
    float16_t golden[N * D];
    float16_t opt[N * D];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_CHECKED(sizeof(struct onnx_node_t));
    node->priv = GenerateLayerNormParam(1e-05f, 0.9f);

    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = N;
    node->inputs[0]->dims[1] = D;
    node->inputs[0]->ndata = node->inputs[0]->dims[0] * node->inputs[0]->dims[1];
    node->inputs[0]->datas = MALLOC_CHECKED(sizeof(float16_t) * node->inputs[0]->ndata);

    float16_t *p = (float16_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = N * D;
    node->outputs[0]->datas = MALLOC_CHECKED(sizeof(float16_t) * node->outputs[0]->ndata);

    BENCH_START(LayerNormalization_float16);
    LayerNormalization_float16(node);
    BENCH_END(LayerNormalization_float16);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(LayerNormalization_float16_rvv);
    LayerNormalization_float16_rvv(node);
    BENCH_END(LayerNormalization_float16_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    ret |= verify_results_f16(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]);
    free(node->outputs);
    FreeLayerNormParam(&node->priv);
    free(node);

    return ret;
}

int test_layernormalization(void)
{
    int ret = 0;
    ret |= test_layernormalization_f16();
    ret |= test_layernormalization_f32();
    return ret;
}