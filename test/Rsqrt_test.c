/**
 * @file Rsqrt_test.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "utils.h"

#define TEST_DATA_LEN 4096

BENCH_DECLARE_VAR()
int test_rsqrt_f32(void)
{
    struct onnx_node_t *node;
    float32_t golden[TEST_DATA_LEN];
    float32_t opt[TEST_DATA_LEN];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 1;

    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = TEST_DATA_LEN;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->inputs[0]->ndata);

    float32_t *p = (float32_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX * 100;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = TEST_DATA_LEN;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->outputs[0]->ndata);

    BENCH_START(Rsqrt_float32);
    Rsqrt_float32(node);
    BENCH_END(Rsqrt_float32);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(Rsqrt_float32_rvv);
    Rsqrt_float32_rvv(node);
    BENCH_END(Rsqrt_float32_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    ret |= verify_results_f32(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_rsqrt_f16(void)
{
    struct onnx_node_t *node;
    float16_t golden[TEST_DATA_LEN];
    float16_t opt[TEST_DATA_LEN];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 1;

    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = TEST_DATA_LEN;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->inputs[0]->ndata);

    float16_t *p = (float16_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX * 100;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = TEST_DATA_LEN;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->outputs[0]->ndata);

    BENCH_START(Rsqrt_float16);
    Rsqrt_float16(node);
    BENCH_END(Rsqrt_float16);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(Rsqrt_float16_rvv);
    Rsqrt_float16_rvv(node);
    BENCH_END(Rsqrt_float16_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    ret |= verify_results_f16(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_rsqrt(void)
{
    int ret = 0;
    ret |= test_rsqrt_f32();
    ret |= test_rsqrt_f16();
    return ret;
}