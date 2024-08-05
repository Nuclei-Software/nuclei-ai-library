/**
 * @file Reduce_test.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-15
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <stdbool.h>

#include "utils.h"

#define DIMS0 (30)
#define DIMS1 (25)
#define DIMS2 (20)

// NOTE: DIMS2 < DIMS1 && DIMS2 < DIMS0 for test

BENCH_DECLARE_VAR()

int test_reduce_all()
{
    struct onnx_node_t *node;
    bool golden[DIMS0 * DIMS1];
    bool *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(bool) * node->inputs[0]->ndata);

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(bool) * node->outputs[0]->ndata);

    // golden test with allaxes true
    node->priv = NULL;
    p = (bool *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = true;
    }
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAll_allaxes_true);
    ReduceAll(node);
    BENCH_END(ReduceAll_allaxes_true);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test with allaxes true
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAll_allaxes_true_rvv);
    ReduceAll_rvv(node);
    BENCH_END(ReduceAll_allaxes_true_rvv);

    // verify result
    ret |= verify_results_int8((int8_t *)golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with allaxes
    p = (bool *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand() % 2;
    }
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAll_allaxes_rand);
    ReduceAll(node);
    BENCH_END(ReduceAll_allaxes_rand);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test with allaxes rand
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAll_allaxes_rand_rvv);
    ReduceAll_rvv(node);
    BENCH_END(ReduceAll_allaxes_rand_rvv);

    // verify result
    ret |= verify_results_int8((int8_t *)golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAll_axis0);
    ReduceAll(node);
    BENCH_END(ReduceAll_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAll_axis0_rvv);
    ReduceAll_rvv(node);
    BENCH_END(ReduceAll_axis0_rvv);

    // verify result
    ret |= verify_results_int8((int8_t *)golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAll_axis1);
    ReduceAll(node);
    BENCH_END(ReduceAll_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAll_axis1_rvv);
    ReduceAll_rvv(node);
    BENCH_END(ReduceAll_axis1_rvv);

    // verify result
    ret |= verify_results_int8((int8_t *)golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAll_axis2);
    ReduceAll(node);
    BENCH_END(ReduceAll_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAll_axis2_rvv);
    ReduceAll_rvv(node);
    BENCH_END(ReduceAll_axis2_rvv);

    // verify result
    ret |= verify_results_int8((int8_t *)golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce_any()
{
    struct onnx_node_t *node;
    bool golden[DIMS0 * DIMS1];
    bool *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(bool) * node->inputs[0]->ndata);

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(bool) * node->outputs[0]->ndata);

    // golden test with allaxes false
    node->priv = NULL;
    p = (bool *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = false;
    }
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAny_allaxes_false);
    ReduceAny(node);
    BENCH_END(ReduceAny_allaxes_false);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test with allaxes false
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAny_allaxes_false_rvv);
    ReduceAny_rvv(node);
    BENCH_END(ReduceAny_allaxes_false_rvv);

    // verify result
    ret |= verify_results_int8((int8_t *)golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with allaxes
    p = (bool *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand() % 2;
    }
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAny_allaxes_rand);
    ReduceAny(node);
    BENCH_END(ReduceAny_allaxes_rand);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test with allaxes rand
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAny_allaxes_rand_rvv);
    ReduceAny_rvv(node);
    BENCH_END(ReduceAny_allaxes_rand_rvv);

    // verify result
    ret |= verify_results_int8((int8_t *)golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAny_axis0);
    ReduceAny(node);
    BENCH_END(ReduceAny_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAny_axis0_rvv);
    ReduceAny_rvv(node);
    BENCH_END(ReduceAny_axis0_rvv);

    // verify result
    ret |= verify_results_int8((int8_t *)golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAny_axis1);
    ReduceAny(node);
    BENCH_END(ReduceAny_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAny_axis1_rvv);
    ReduceAny_rvv(node);
    BENCH_END(ReduceAny_axis1_rvv);

    // verify result
    ret |= verify_results_int8((int8_t *)golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAny_axis1);
    ReduceAny(node);
    BENCH_END(ReduceAny_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceAny_axis2_rvv);
    ReduceAny_rvv(node);
    BENCH_END(ReduceAny_axis2_rvv);

    // verify result
    ret |= verify_results_int8((int8_t *)golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce_max_int8()
{
    struct onnx_node_t *node;
    int8_t golden[DIMS0 * DIMS1];
    int8_t *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(int8_t) * node->inputs[0]->ndata);
    p = (int8_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(int8_t) * node->outputs[0]->ndata);

    // golden test with allaxes
    node->priv = NULL;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int8_allaxes);
    ReduceMax_int8(node);
    BENCH_END(ReduceMax_int8_allaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test with allaxes
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int8_allaxes_rvv);
    ReduceMax_int8_rvv(node);
    BENCH_END(ReduceMax_int8_allaxes_rvv);

    // verify result
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int8_axis0);
    ReduceMax_int8(node);
    BENCH_END(ReduceMax_int8_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int8_axis0_rvv);
    ReduceMax_int8_rvv(node);
    BENCH_END(ReduceMax_int8_axis0_rvv);

    // verify result
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int8_axis1);
    ReduceMax_int8(node);
    BENCH_END(ReduceMax_int8_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int8_axis1_rvv);
    ReduceMax_int8_rvv(node);
    BENCH_END(ReduceMax_int8_axis1_rvv);

    // verify result
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int8_axis1);
    ReduceMax_int8(node);
    BENCH_END(ReduceMax_int8_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int8_axis2_rvv);
    ReduceMax_int8_rvv(node);
    BENCH_END(ReduceMax_int8_axis2_rvv);

    // verify result
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce_max_float16()
{
    struct onnx_node_t *node;
    float16_t golden[DIMS0 * DIMS1];
    float16_t *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(float16_t) * node->inputs[0]->ndata);
    p = (float16_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(float16_t) * node->outputs[0]->ndata);

    // golden test with allaxes
    node->priv = NULL;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float16_allaxes);
    ReduceMax_float16(node);
    BENCH_END(ReduceMax_float16_allaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    // rvv optimization test with allaxes
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float16_allaxes_rvv);
    ReduceMax_float16_rvv(node);
    BENCH_END(ReduceMax_float16_allaxes_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float16_axis0);
    ReduceMax_float16(node);
    BENCH_END(ReduceMax_float16_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float16_axis0_rvv);
    ReduceMax_float16_rvv(node);
    BENCH_END(ReduceMax_float16_axis0_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float16_axis1);
    ReduceMax_float16(node);
    BENCH_END(ReduceMax_float16_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float16_axis1_rvv);
    ReduceMax_float16_rvv(node);
    BENCH_END(ReduceMax_float16_axis1_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float16_axis1);
    ReduceMax_float16(node);
    BENCH_END(ReduceMax_float16_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float16_axis2_rvv);
    ReduceMax_float16_rvv(node);
    BENCH_END(ReduceMax_float16_axis2_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce_max_float32()
{
    struct onnx_node_t *node;
    float32_t golden[DIMS0 * DIMS1];
    float32_t *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(float32_t) * node->inputs[0]->ndata);
    p = (float32_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(float32_t) * node->outputs[0]->ndata);

    // golden test with allaxes
    node->priv = NULL;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float32_allaxes);
    ReduceMax_float32(node);
    BENCH_END(ReduceMax_float32_allaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    // rvv optimization test with allaxes
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float32_allaxes_rvv);
    ReduceMax_float32_rvv(node);
    BENCH_END(ReduceMax_float32_allaxes_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float32_axis0);
    ReduceMax_float32(node);
    BENCH_END(ReduceMax_float32_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float32_axis0_rvv);
    ReduceMax_float32_rvv(node);
    BENCH_END(ReduceMax_float32_axis0_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float32_axis1);
    ReduceMax_float32(node);
    BENCH_END(ReduceMax_float32_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float32_axis1_rvv);
    ReduceMax_float32_rvv(node);
    BENCH_END(ReduceMax_float32_axis1_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float32_axis1);
    ReduceMax_float32(node);
    BENCH_END(ReduceMax_float32_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_float32_axis2_rvv);
    ReduceMax_float32_rvv(node);
    BENCH_END(ReduceMax_float32_axis2_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce_max_int32()
{
    struct onnx_node_t *node;
    int32_t golden[DIMS0 * DIMS1];
    int32_t *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(int32_t) * node->inputs[0]->ndata);
    p = (int32_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(int32_t) * node->outputs[0]->ndata);

    // golden test with allaxes
    node->priv = NULL;
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int32_allaxes);
    ReduceMax_int32(node);
    BENCH_END(ReduceMax_int32_allaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    // rvv optimization test with allaxes
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int32_allaxes_rvv);
    ReduceMax_int32_rvv(node);
    BENCH_END(ReduceMax_int32_allaxes_rvv);

    // verify result
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int32_axis0);
    ReduceMax_int32(node);
    BENCH_END(ReduceMax_int32_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int32_axis0_rvv);
    ReduceMax_int32_rvv(node);
    BENCH_END(ReduceMax_int32_axis0_rvv);

    // verify result
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int32_axis1);
    ReduceMax_int32(node);
    BENCH_END(ReduceMax_int32_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int32_axis1_rvv);
    ReduceMax_int32_rvv(node);
    BENCH_END(ReduceMax_int32_axis1_rvv);

    // verify result
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int32_axis1);
    ReduceMax_int32(node);
    BENCH_END(ReduceMax_int32_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMax_int32_axis2_rvv);
    ReduceMax_int32_rvv(node);
    BENCH_END(ReduceMax_int32_axis2_rvv);

    // verify result
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce_min_int8()
{
    struct onnx_node_t *node;
    int8_t golden[DIMS0 * DIMS1];
    int8_t *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(int8_t) * node->inputs[0]->ndata);
    p = (int8_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(int8_t) * node->outputs[0]->ndata);

    // golden test with allaxes
    node->priv = NULL;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int8_allaxes);
    ReduceMin_int8(node);
    BENCH_END(ReduceMin_int8_allaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test with allaxes
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int8_allaxes_rvv);
    ReduceMin_int8_rvv(node);
    BENCH_END(ReduceMin_int8_allaxes_rvv);

    // verify result
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int8_axis0);
    ReduceMin_int8(node);
    BENCH_END(ReduceMin_int8_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int8_axis0_rvv);
    ReduceMin_int8_rvv(node);
    BENCH_END(ReduceMin_int8_axis0_rvv);

    // verify result
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int8_axis1);
    ReduceMin_int8(node);
    BENCH_END(ReduceMin_int8_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int8_axis1_rvv);
    ReduceMin_int8_rvv(node);
    BENCH_END(ReduceMin_int8_axis1_rvv);

    // verify result
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int8_axis1);
    ReduceMin_int8(node);
    BENCH_END(ReduceMin_int8_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int8_axis2_rvv);
    ReduceMin_int8_rvv(node);
    BENCH_END(ReduceMin_int8_axis2_rvv);

    // verify result
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce_min_float16()
{
    struct onnx_node_t *node;
    float16_t golden[DIMS0 * DIMS1];
    float16_t *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(float16_t) * node->inputs[0]->ndata);
    p = (float16_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(float16_t) * node->outputs[0]->ndata);

    // golden test with allaxes
    node->priv = NULL;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float16_allaxes);
    ReduceMin_float16(node);
    BENCH_END(ReduceMin_float16_allaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    // rvv optimization test with allaxes
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float16_allaxes_rvv);
    ReduceMin_float16_rvv(node);
    BENCH_END(ReduceMin_float16_allaxes_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float16_axis0);
    ReduceMin_float16(node);
    BENCH_END(ReduceMin_float16_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float16_axis0_rvv);
    ReduceMin_float16_rvv(node);
    BENCH_END(ReduceMin_float16_axis0_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float16_axis1);
    ReduceMin_float16(node);
    BENCH_END(ReduceMin_float16_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float16_axis1_rvv);
    ReduceMin_float16_rvv(node);
    BENCH_END(ReduceMin_float16_axis1_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float16_axis1);
    ReduceMin_float16(node);
    BENCH_END(ReduceMin_float16_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float16_axis2_rvv);
    ReduceMin_float16_rvv(node);
    BENCH_END(ReduceMin_float16_axis2_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce_min_float32()
{
    struct onnx_node_t *node;
    float32_t golden[DIMS0 * DIMS1];
    float32_t *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(float32_t) * node->inputs[0]->ndata);
    p = (float32_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(float32_t) * node->outputs[0]->ndata);

    // golden test with allaxes
    node->priv = NULL;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float32_allaxes);
    ReduceMin_float32(node);
    BENCH_END(ReduceMin_float32_allaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    // rvv optimization test with allaxes
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float32_allaxes_rvv);
    ReduceMin_float32_rvv(node);
    BENCH_END(ReduceMin_float32_allaxes_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float32_axis0);
    ReduceMin_float32(node);
    BENCH_END(ReduceMin_float32_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float32_axis0_rvv);
    ReduceMin_float32_rvv(node);
    BENCH_END(ReduceMin_float32_axis0_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float32_axis1);
    ReduceMin_float32(node);
    BENCH_END(ReduceMin_float32_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float32_axis1_rvv);
    ReduceMin_float32_rvv(node);
    BENCH_END(ReduceMin_float32_axis1_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float32_axis1);
    ReduceMin_float32(node);
    BENCH_END(ReduceMin_float32_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_float32_axis2_rvv);
    ReduceMin_float32_rvv(node);
    BENCH_END(ReduceMin_float32_axis2_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce_min_int32()
{
    struct onnx_node_t *node;
    int32_t golden[DIMS0 * DIMS1];
    int32_t *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(int32_t) * node->inputs[0]->ndata);
    p = (int32_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(int32_t) * node->outputs[0]->ndata);

    // golden test with allaxes
    node->priv = NULL;
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int32_allaxes);
    ReduceMin_int32(node);
    BENCH_END(ReduceMin_int32_allaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    // rvv optimization test with allaxes
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int32_allaxes_rvv);
    ReduceMin_int32_rvv(node);
    BENCH_END(ReduceMin_int32_allaxes_rvv);

    // verify result
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int32_axis0);
    ReduceMin_int32(node);
    BENCH_END(ReduceMin_int32_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int32_axis0_rvv);
    ReduceMin_int32_rvv(node);
    BENCH_END(ReduceMin_int32_axis0_rvv);

    // verify result
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int32_axis1);
    ReduceMin_int32(node);
    BENCH_END(ReduceMin_int32_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int32_axis1_rvv);
    ReduceMin_int32_rvv(node);
    BENCH_END(ReduceMin_int32_axis1_rvv);

    // verify result
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int32_axis1);
    ReduceMin_int32(node);
    BENCH_END(ReduceMin_int32_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceMin_int32_axis2_rvv);
    ReduceMin_int32_rvv(node);
    BENCH_END(ReduceMin_int32_axis2_rvv);

    // verify result
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce_sum_float16()
{
    struct onnx_node_t *node;
    float16_t golden[DIMS0 * DIMS1];
    float16_t *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(float16_t) * node->inputs[0]->ndata);
    p = (float16_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand() * 1.0 / rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(float16_t) * node->outputs[0]->ndata);

    // golden test with allaxes
    node->priv = NULL;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float16_allaxes);
    ReduceSum_float16(node);
    BENCH_END(ReduceSum_float16_allaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    // rvv optimization test with allaxes
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float16_allaxes_rvv);
    ReduceSum_float16_rvv(node);
    BENCH_END(ReduceSum_float16_allaxes_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float16_axis0);
    ReduceSum_float16(node);
    BENCH_END(ReduceSum_float16_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float16_axis0_rvv);
    ReduceSum_float16_rvv(node);
    BENCH_END(ReduceSum_float16_axis0_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float16_axis1);
    ReduceSum_float16(node);
    BENCH_END(ReduceSum_float16_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float16_axis1_rvv);
    ReduceSum_float16_rvv(node);
    BENCH_END(ReduceSum_float16_axis1_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float16_axis1);
    ReduceSum_float16(node);
    BENCH_END(ReduceSum_float16_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float16_axis2_rvv);
    ReduceSum_float16_rvv(node);
    BENCH_END(ReduceSum_float16_axis2_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce_sum_float32()
{
    struct onnx_node_t *node;
    float32_t golden[DIMS0 * DIMS1];
    float32_t *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(float32_t) * node->inputs[0]->ndata);
    p = (float32_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand() * 1.0 / rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(float32_t) * node->outputs[0]->ndata);

    // golden test with allaxes
    node->priv = NULL;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float32_allaxes);
    ReduceSum_float32(node);
    BENCH_END(ReduceSum_float32_allaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    // rvv optimization test with allaxes
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float32_allaxes_rvv);
    ReduceSum_float32_rvv(node);
    BENCH_END(ReduceSum_float32_allaxes_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float32_axis0);
    ReduceSum_float32(node);
    BENCH_END(ReduceSum_float32_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float32_axis0_rvv);
    ReduceSum_float32_rvv(node);
    BENCH_END(ReduceSum_float32_axis0_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float32_axis1);
    ReduceSum_float32(node);
    BENCH_END(ReduceSum_float32_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float32_axis1_rvv);
    ReduceSum_float32_rvv(node);
    BENCH_END(ReduceSum_float32_axis1_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float32_axis1);
    ReduceSum_float32(node);
    BENCH_END(ReduceSum_float32_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceSum_float32_axis2_rvv);
    ReduceSum_float32_rvv(node);
    BENCH_END(ReduceSum_float32_axis2_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce_prod_float16()
{
    struct onnx_node_t *node;
    float16_t golden[DIMS0 * DIMS1];
    float16_t *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(float16_t) * node->inputs[0]->ndata);
    p = (float16_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = 1.1f;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(float16_t) * node->outputs[0]->ndata);

    // golden test with allaxes
    node->priv = NULL;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float16_allaxes);
    ReduceProd_float16(node);
    BENCH_END(ReduceProd_float16_allaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    // rvv optimization test with allaxes
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float16_allaxes_rvv);
    ReduceProd_float16_rvv(node);
    BENCH_END(ReduceProd_float16_allaxes_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float16_axis0);
    ReduceProd_float16(node);
    BENCH_END(ReduceProd_float16_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float16_axis0_rvv);
    ReduceProd_float16_rvv(node);
    BENCH_END(ReduceProd_float16_axis0_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float16_axis1);
    ReduceProd_float16(node);
    BENCH_END(ReduceProd_float16_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float16_axis1_rvv);
    ReduceProd_float16_rvv(node);
    BENCH_END(ReduceProd_float16_axis1_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float16_axis1);
    ReduceProd_float16(node);
    BENCH_END(ReduceProd_float16_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float16_axis2_rvv);
    ReduceProd_float16_rvv(node);
    BENCH_END(ReduceProd_float16_axis2_rvv);

    // verify result
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce_prod_float32()
{
    struct onnx_node_t *node;
    float32_t golden[DIMS0 * DIMS1];
    float32_t *p;
    int ret = 0;
    int axis;

    node = (struct onnx_node_t *)malloc(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = DIMS0 * DIMS1 * DIMS2;
    node->inputs[0]->ndim = 3;
    node->inputs[0]->dims = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = DIMS0;
    node->inputs[0]->dims[1] = DIMS1;
    node->inputs[0]->dims[2] = DIMS2;
    node->inputs[0]->strides = (int *)malloc(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = DIMS0;
    node->inputs[0]->strides[2] = DIMS0 * DIMS1;
    node->inputs[0]->datas = malloc(sizeof(float32_t) * node->inputs[0]->ndata);
    p = (float32_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = 1.1f;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)malloc(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)malloc(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = DIMS0 * DIMS1;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = DIMS0;
    node->outputs[0]->dims[1] = DIMS1;
    node->outputs[0]->strides = (int *)malloc(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = DIMS0;
    node->outputs[0]->datas = malloc(sizeof(float32_t) * node->outputs[0]->ndata);

    // golden test with allaxes
    node->priv = NULL;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float32_allaxes);
    ReduceProd_float32(node);
    BENCH_END(ReduceProd_float32_allaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    // rvv optimization test with allaxes
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float32_allaxes_rvv);
    ReduceProd_float32_rvv(node);
    BENCH_END(ReduceProd_float32_allaxes_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    node->priv = &axis;
    axis = 0;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float32_axis0);
    ReduceProd_float32(node);
    BENCH_END(ReduceProd_float32_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float32_axis0_rvv);
    ReduceProd_float32_rvv(node);
    BENCH_END(ReduceProd_float32_axis0_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 1;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float32_axis1);
    ReduceProd_float32(node);
    BENCH_END(ReduceProd_float32_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float32_axis1_rvv);
    ReduceProd_float32_rvv(node);
    BENCH_END(ReduceProd_float32_axis1_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    axis = 2;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float32_axis1);
    ReduceProd_float32(node);
    BENCH_END(ReduceProd_float32_axis2);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ReduceProd_float32_axis2_rvv);
    ReduceProd_float32_rvv(node);
    BENCH_END(ReduceProd_float32_axis2_rvv);

    // verify result
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->strides);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->strides);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->outputs);
    free(node);

    return ret;
}

int test_reduce(void)
{
    int ret = 0;
    ret |= test_reduce_all();
    ret |= test_reduce_any();
    ret |= test_reduce_max_int8();
    ret |= test_reduce_max_int32();
    ret |= test_reduce_max_float16();
    ret |= test_reduce_max_float32();
    ret |= test_reduce_min_int8();
    ret |= test_reduce_min_int32();
    ret |= test_reduce_min_float16();
    ret |= test_reduce_min_float32();
    ret |= test_reduce_sum_float16();
    ret |= test_reduce_sum_float32();
    ret |= test_reduce_prod_float16();
    ret |= test_reduce_prod_float32();
    return ret;
}