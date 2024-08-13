/**
 * @file BatchNorm_test.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "utils.h"

#define DIM0 1
#define DIM1 3
#define DIM2 8
#define DIM3 256

BENCH_DECLARE_VAR()
int test_batchnormalization_f32(void)
{
    struct onnx_node_t *node;
    float32_t golden[DIM0 * DIM1 * DIM2 * DIM3];
    float32_t opt[DIM0 * DIM1 * DIM2 * DIM3];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_CHECKED(sizeof(struct onnx_node_t));
    node->priv = GenerateBatchNormParam(1e-05f, 0.9f);

    node->ninput = 5;
    node->inputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndim = 4;
    node->inputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIM0;
    node->inputs[0]->dims[1] = DIM1;
    node->inputs[0]->dims[2] = DIM2;
    node->inputs[0]->dims[3] = DIM3;
    node->inputs[0]->ndata = node->inputs[0]->dims[0] * node->inputs[0]->dims[1] * node->inputs[0]->dims[2] * node->inputs[0]->dims[3];
    node->inputs[0]->datas = MALLOC_CHECKED(sizeof(float32_t) * node->inputs[0]->ndata);

    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndim = 1;
    node->inputs[1]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[1]->ndim);
    node->inputs[1]->dims[0] = DIM1;
    node->inputs[1]->ndata = node->inputs[1]->dims[0];
    node->inputs[1]->datas = MALLOC_CHECKED(sizeof(float32_t) * node->inputs[1]->ndata);

    node->inputs[2] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[2]->ndim = 1;
    node->inputs[2]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[2]->ndim);
    node->inputs[2]->dims[0] = DIM1;
    node->inputs[2]->ndata = node->inputs[2]->dims[0];
    node->inputs[2]->datas = MALLOC_CHECKED(sizeof(float32_t) * node->inputs[2]->ndata);

    node->inputs[3] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[3]->ndim = 1;
    node->inputs[3]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[3]->ndim);
    node->inputs[3]->dims[0] = DIM1;
    node->inputs[3]->ndata = node->inputs[3]->dims[0];
    node->inputs[3]->datas = MALLOC_CHECKED(sizeof(float32_t) * node->inputs[3]->ndata);

    node->inputs[4] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[4]->ndim = 1;
    node->inputs[4]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[4]->ndim);
    node->inputs[4]->dims[0] = DIM1;
    node->inputs[4]->ndata = node->inputs[4]->dims[0];
    node->inputs[4]->datas = MALLOC_CHECKED(sizeof(float32_t) * node->inputs[4]->ndata);

    float32_t *p = (float32_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }
    p = (float32_t *)node->inputs[1]->datas;
    for (int i = 0; i < node->inputs[1]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }
    p = (float32_t *)node->inputs[2]->datas;
    for (int i = 0; i < node->inputs[2]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }
    p = (float32_t *)node->inputs[3]->datas;
    for (int i = 0; i < node->inputs[3]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }
    p = (float32_t *)node->inputs[4]->datas;
    for (int i = 0; i < node->inputs[4]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIM0 * DIM1 * DIM2 * DIM3;
    node->outputs[0]->datas = MALLOC_CHECKED(sizeof(float32_t) * node->outputs[0]->ndata);

    BENCH_START(BatchNormalization_float32);
    BatchNormalization_float32(node);
    BENCH_END(BatchNormalization_float32);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(BatchNormalization_float32_rvv);
    BatchNormalization_float32_rvv(node);
    BENCH_END(BatchNormalization_float32_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    ret |= verify_results_f32(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[1]->datas);
    free(node->inputs[2]->datas);
    free(node->inputs[3]->datas);
    free(node->inputs[4]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[1]->dims);
    free(node->inputs[2]->dims);
    free(node->inputs[3]->dims);
    free(node->inputs[4]->dims);
    free(node->inputs[0]);
    free(node->inputs[1]);
    free(node->inputs[2]);
    free(node->inputs[3]);
    free(node->inputs[4]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]);
    free(node->outputs);
    FreeBatchNormParam(&node->priv);
    free(node);

    return ret;
}

int test_batchnormalization_f16(void)
{
    struct onnx_node_t *node;
    float16_t golden[DIM0 * DIM1 * DIM2 * DIM3];
    float16_t opt[DIM0 * DIM1 * DIM2 * DIM3];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_CHECKED(sizeof(struct onnx_node_t));
    node->priv = GenerateBatchNormParam(1e-05f, 0.9f);

    node->ninput = 5;
    node->inputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndim = 4;
    node->inputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIM0;
    node->inputs[0]->dims[1] = DIM1;
    node->inputs[0]->dims[2] = DIM2;
    node->inputs[0]->dims[3] = DIM3;
    node->inputs[0]->ndata = node->inputs[0]->dims[0] * node->inputs[0]->dims[1] * node->inputs[0]->dims[2] * node->inputs[0]->dims[3];
    node->inputs[0]->datas = MALLOC_CHECKED(sizeof(float16_t) * node->inputs[0]->ndata);

    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndim = 1;
    node->inputs[1]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[1]->ndim);
    node->inputs[1]->dims[0] = DIM1;
    node->inputs[1]->ndata = node->inputs[1]->dims[0];
    node->inputs[1]->datas = MALLOC_CHECKED(sizeof(float16_t) * node->inputs[1]->ndata);

    node->inputs[2] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[2]->ndim = 1;
    node->inputs[2]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[2]->ndim);
    node->inputs[2]->dims[0] = DIM1;
    node->inputs[2]->ndata = node->inputs[2]->dims[0];
    node->inputs[2]->datas = MALLOC_CHECKED(sizeof(float16_t) * node->inputs[2]->ndata);

    node->inputs[3] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[3]->ndim = 1;
    node->inputs[3]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[3]->ndim);
    node->inputs[3]->dims[0] = DIM1;
    node->inputs[3]->ndata = node->inputs[3]->dims[0];
    node->inputs[3]->datas = MALLOC_CHECKED(sizeof(float16_t) * node->inputs[3]->ndata);

    node->inputs[4] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->inputs[4]->ndim = 1;
    node->inputs[4]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[4]->ndim);
    node->inputs[4]->dims[0] = DIM1;
    node->inputs[4]->ndata = node->inputs[4]->dims[0];
    node->inputs[4]->datas = MALLOC_CHECKED(sizeof(float16_t) * node->inputs[4]->ndata);

    float16_t *p = (float16_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }
    p = (float16_t *)node->inputs[1]->datas;
    for (int i = 0; i < node->inputs[1]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }
    p = (float16_t *)node->inputs[2]->datas;
    for (int i = 0; i < node->inputs[2]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }
    p = (float16_t *)node->inputs[3]->datas;
    for (int i = 0; i < node->inputs[3]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }
    p = (float16_t *)node->inputs[4]->datas;
    for (int i = 0; i < node->inputs[4]->ndata; i++) {
        p[i] = rand() * 1.0 / RAND_MAX;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIM0 * DIM1 * DIM2 * DIM3;
    node->outputs[0]->datas = MALLOC_CHECKED(sizeof(float16_t) * node->outputs[0]->ndata);

    BENCH_START(BatchNormalization_float16);
    BatchNormalization_float16(node);
    BENCH_END(BatchNormalization_float16);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(BatchNormalization_float16_rvv);
    BatchNormalization_float16_rvv(node);
    BENCH_END(BatchNormalization_float16_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    ret |= verify_results_f16(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[1]->datas);
    free(node->inputs[2]->datas);
    free(node->inputs[3]->datas);
    free(node->inputs[4]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[1]->dims);
    free(node->inputs[2]->dims);
    free(node->inputs[3]->dims);
    free(node->inputs[4]->dims);
    free(node->inputs[0]);
    free(node->inputs[1]);
    free(node->inputs[2]);
    free(node->inputs[3]);
    free(node->inputs[4]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]);
    free(node->outputs);
    FreeBatchNormParam(&node->priv);
    free(node);

    return ret;
}

int test_batchnormalization(void)
{
    int ret = 0;
    ret = test_batchnormalization_f16();
    ret = test_batchnormalization_f32();
    return ret;
}