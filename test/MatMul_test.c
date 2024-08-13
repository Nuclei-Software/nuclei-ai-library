/**
 * @file Matmul_test.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "utils.h"

#define M 128
#define N 128
#define K 128

BENCH_DECLARE_VAR()
int test_matmul_int8(void)
{
    struct onnx_node_t *node;
    int8_t golden[M * N];
    int8_t opt[M * N];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_CHECKED(sizeof(struct onnx_node_t));
    node->priv = NULL;
    node->ninput = 2;

    node->inputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = M * K;
    node->inputs[0]->datas = MALLOC_CHECKED(sizeof(int8_t) * node->inputs[0]->ndata);
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = K;
    node->inputs[0]->dims[1] = M;
    int8_t *p = (int8_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand();
    }

    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = K * N;
    node->inputs[1]->datas = MALLOC_CHECKED(sizeof(int8_t) * node->inputs[1]->ndata);
    node->inputs[1]->ndim = 2;
    node->inputs[1]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[1]->ndim);
    node->inputs[1]->dims[0] = N;
    node->inputs[1]->dims[1] = K;
    p = (int8_t *)node->inputs[1]->datas;
    for (int i = 0; i < node->inputs[1]->ndata; i++) {
        p[i] = rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = M * N;
    node->outputs[0]->datas = MALLOC_CHECKED(sizeof(int8_t) * node->outputs[0]->ndata);
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = N;
    node->outputs[0]->dims[1] = M;

    BENCH_START(MatMul_int8);
    MatMul_int8(node);
    BENCH_END(MatMul_int8);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int8_t));
    BENCH_START(MatMul_int8_rvv);
    MatMul_int8_rvv(node);
    BENCH_END(MatMul_int8_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    ret |= verify_results_int8(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs[1]->datas);
    free(node->inputs[1]->dims);
    free(node->inputs[1]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_matmul_f16(void)
{
    struct onnx_node_t *node;
    float16_t golden[M * N];
    float16_t opt[M * N];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_CHECKED(sizeof(struct onnx_node_t));
    node->priv = NULL;
    node->ninput = 2;

    node->inputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = M * K;
    node->inputs[0]->datas = MALLOC_CHECKED(sizeof(float16_t) * node->inputs[0]->ndata);
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = K;
    node->inputs[0]->dims[1] = M;
    float16_t *p = (float16_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }

    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = K * N;
    node->inputs[1]->datas = MALLOC_CHECKED(sizeof(float16_t) * node->inputs[1]->ndata);
    node->inputs[1]->ndim = 2;
    node->inputs[1]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[1]->ndim);
    node->inputs[1]->dims[0] = N;
    node->inputs[1]->dims[1] = K;
    p = (float16_t *)node->inputs[1]->datas;
    for (int i = 0; i < node->inputs[1]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = M * N;
    node->outputs[0]->datas = MALLOC_CHECKED(sizeof(float16_t) * node->outputs[0]->ndata);
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = N;
    node->outputs[0]->dims[1] = M;

    BENCH_START(MatMul_float16);
    MatMul_float16(node);
    BENCH_END(MatMul_float16);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(MatMul_float16_rvv);
    MatMul_float16_rvv(node);
    BENCH_END(MatMul_float16_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    ret |= verify_results_f16(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs[1]->datas);
    free(node->inputs[1]->dims);
    free(node->inputs[1]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_matmul_f32(void)
{
    struct onnx_node_t *node;
    float32_t golden[M * N];
    float32_t opt[M * N];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_CHECKED(sizeof(struct onnx_node_t));
    node->ninput = 2;

    node->inputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = M * K;
    node->inputs[0]->datas = MALLOC_CHECKED(sizeof(float32_t) * node->inputs[0]->ndata);
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = K;
    node->inputs[0]->dims[1] = M;
    float32_t *p = (float32_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }

    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = K * N;
    node->inputs[1]->datas = MALLOC_CHECKED(sizeof(float32_t) * node->inputs[1]->ndata);
    node->inputs[1]->ndim = 2;
    node->inputs[1]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[1]->ndim);
    node->inputs[1]->dims[0] = N;
    node->inputs[1]->dims[1] = K;
    p = (float32_t *)node->inputs[1]->datas;
    for (int i = 0; i < node->inputs[1]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = M * N;
    node->outputs[0]->datas = MALLOC_CHECKED(sizeof(float32_t) * node->outputs[0]->ndata);
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = N;
    node->outputs[0]->dims[1] = M;

    BENCH_START(MatMul_float32);
    MatMul_float32(node);
    BENCH_END(MatMul_float32);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(MatMul_float32_rvv);
    MatMul_float32_rvv(node);
    BENCH_END(MatMul_float32_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    ret |= verify_results_f32(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs[1]->datas);
    free(node->inputs[1]->dims);
    free(node->inputs[1]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_matmul(void)
{
    int ret = 0;
    ret |= test_matmul_int8();
    ret |= test_matmul_f16();
    ret |= test_matmul_f32();
    return ret;
}