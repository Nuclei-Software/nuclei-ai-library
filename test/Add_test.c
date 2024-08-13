/**
 * @file Add_test.c
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
int test_add_int8(void)
{
    struct onnx_node_t *node;
    int8_t golden[TEST_DATA_LEN];
    int8_t opt[TEST_DATA_LEN];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 2;

    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = TEST_DATA_LEN;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(int8_t) * node->inputs[0]->ndata);

    int8_t *p = (int8_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand();
    }
    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = TEST_DATA_LEN;
    node->inputs[1]->datas = MALLOC_ASSERT(sizeof(int8_t) * node->inputs[1]->ndata);
    p = (int8_t *)node->inputs[1]->datas;
    for (int i = 0; i < node->inputs[1]->ndata; i++) {
        p[i] = rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = TEST_DATA_LEN;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(int8_t) * node->outputs[0]->ndata);

    BENCH_START(Add_int8);
    Add_int8(node);
    BENCH_END(Add_int8);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int8_t));
    BENCH_START(Add_int8_rvv);
    Add_int8_rvv(node);
    BENCH_END(Add_int8_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    ret |= verify_results_int8(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]);
    free(node->inputs[1]->datas);
    free(node->inputs[1]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_add_f16(void)
{
    struct onnx_node_t *node;
    float16_t golden[TEST_DATA_LEN];
    float16_t opt[TEST_DATA_LEN];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 2;

    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = TEST_DATA_LEN;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->inputs[0]->ndata);

    float16_t *p = (float16_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }
    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = TEST_DATA_LEN;
    node->inputs[1]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->inputs[1]->ndata);
    p = (float16_t *)node->inputs[1]->datas;
    for (int i = 0; i < node->inputs[1]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = TEST_DATA_LEN;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->outputs[0]->ndata);

    BENCH_START(Add_float16);
    Add_float16(node);
    BENCH_END(Add_float16);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(Add_float16_rvv);
    Add_float16_rvv(node);
    BENCH_END(Add_float16_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    ret |= verify_results_f16(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]);
    free(node->inputs[1]->datas);
    free(node->inputs[1]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_add_f32(void)
{
    struct onnx_node_t *node;
    float32_t golden[TEST_DATA_LEN];
    float32_t opt[TEST_DATA_LEN];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 2;

    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = TEST_DATA_LEN;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->inputs[0]->ndata);

    float32_t *p = (float32_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }
    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = TEST_DATA_LEN;
    node->inputs[1]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->inputs[1]->ndata);
    p = (float32_t *)node->inputs[1]->datas;
    for (int i = 0; i < node->inputs[1]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = TEST_DATA_LEN;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->outputs[0]->ndata);

    BENCH_START(Add_float32);
    Add_float32(node);
    BENCH_END(Add_float32);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(Add_float32_rvv);
    Add_float32_rvv(node);
    BENCH_END(Add_float32_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    ret |= verify_results_f32(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]);
    free(node->inputs[1]->datas);
    free(node->inputs[1]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);
    return ret;
}

int test_add(void)
{
    int ret = 0;
    ret |= test_add_int8();
    ret |= test_add_f16();
    ret |= test_add_f32();
    return ret;
}