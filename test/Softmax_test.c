/**
 * @file Softmax_test.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "utils.h"

#define NUM_ROWS 4
#define NUM_COLS 512

BENCH_DECLARE_VAR()
int test_softmax_f32(void)
{
    struct onnx_node_t *node;
    float32_t golden[NUM_ROWS * NUM_COLS];
    float32_t opt[NUM_ROWS * NUM_COLS];
    int ret = 0;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->priv = NULL;
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = NUM_COLS;
    node->inputs[0]->dims[1] = NUM_ROWS;
    node->inputs[0]->ndata = NUM_ROWS * NUM_COLS;
    node->inputs[0]->datas = malloc(sizeof(float32_t) * node->inputs[0]->ndata);

    float32_t *p = (float32_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = NUM_COLS;
    node->outputs[0]->dims[1] = NUM_ROWS;
    node->outputs[0]->ndata = NUM_ROWS * NUM_COLS;
    node->outputs[0]->datas = malloc(sizeof(float32_t) * node->outputs[0]->ndata);

    BENCH_START(Softmax_float32);
    Softmax_float32(node);
    BENCH_END(Softmax_float32);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(Softmax_float32_rvv);
    Softmax_float32_rvv(node);
    BENCH_END(Softmax_float32_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    ret |= verify_results_f32(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_softmax_f16(void)
{
    struct onnx_node_t *node;
    float16_t golden[NUM_ROWS * NUM_COLS];
    float16_t opt[NUM_ROWS * NUM_COLS];
    int ret = 0;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->priv = NULL;
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = NUM_COLS;
    node->inputs[0]->dims[1] = NUM_ROWS;
    node->inputs[0]->ndata = NUM_COLS * NUM_ROWS;
    node->inputs[0]->datas = malloc(sizeof(float16_t) * node->inputs[0]->ndata);

    float16_t *p = (float16_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = NUM_COLS;
    node->outputs[0]->dims[1] = NUM_ROWS;
    node->outputs[0]->ndata = NUM_COLS * NUM_ROWS;
    node->outputs[0]->datas = malloc(sizeof(float16_t) * node->outputs[0]->ndata);

    BENCH_START(Softmax_float16);
    Softmax_float16(node);
    BENCH_END(Softmax_float16);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(Softmax_float16_rvv);
    Softmax_float16_rvv(node);
    BENCH_END(Softmax_float16_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    ret |= verify_results_f16(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_softmax(void)
{
    int ret = 0;
    ret |= test_softmax_f32();
    ret |= test_softmax_f16();
    return ret;
}