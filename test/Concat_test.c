/**
 * @file Concat_test.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-09
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "utils.h"

#define SIZE 32
#define INPUT_TENSORS 3

BENCH_DECLARE_VAR()

int test_concat_int8()
{
    struct onnx_node_t *node;
    int8_t golden[SIZE * SIZE * INPUT_TENSORS];
    int axis = 0;
    int8_t *p;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_CHECKED(sizeof(struct onnx_node_t));
    node->ninput = INPUT_TENSORS;
    node->inputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->ninput);

    for (int i = 0; i < INPUT_TENSORS; ++i) {
        node->inputs[i] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
        node->inputs[i]->ndata = SIZE * SIZE;
        node->inputs[i]->ndim = 2;
        node->inputs[i]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[i]->ndim);
        node->inputs[i]->dims[0] = SIZE;
        node->inputs[i]->dims[1] = SIZE;
        node->inputs[i]->datas = MALLOC_CHECKED(sizeof(int8_t) * node->inputs[i]->ndata);
        p = (int8_t *)node->inputs[i]->datas;
        for (int j = 0; j < node->inputs[0]->ndata; j++) {
            p[j] = rand();
        }
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = SIZE * SIZE * INPUT_TENSORS;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE;
    node->outputs[0]->dims[1] = SIZE * INPUT_TENSORS;
    node->outputs[0]->datas = MALLOC_CHECKED(sizeof(int8_t) * node->outputs[0]->ndata);
    node->priv = &axis;

    // golden test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(Concat_int8_axis0);
    Concat_int8(node);
    BENCH_END(Concat_int8_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test with axis = 0
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int8_t));
    BENCH_START(Concat_int8_rvv_axis0);
    Concat_int8_rvv(node);
    BENCH_END(Concat_int8_rvv_axis0);

    // verify result with axis = 0
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with axis = 1
    axis = 1;
    node->outputs[0]->dims[0] = SIZE * INPUT_TENSORS;
    node->outputs[0]->dims[1] = SIZE;
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int8_t));
    BENCH_START(Concat_int8_axis1);
    Concat_int8(node);
    BENCH_END(Concat_int8_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test with axis = 1
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int8_t));
    BENCH_START(Concat_int8_rvv_axis1);
    Concat_int8_rvv(node);
    BENCH_END(Concat_int8_rvv_axis1);

    // verify result with axis = 1
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    for (int i = 0; i < node->ninput; ++i) {
        free(node->inputs[i]->datas);
        free(node->inputs[i]->dims);
        free(node->inputs[i]);
    }
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_concat_int32()
{
    struct onnx_node_t *node;
    int32_t golden[SIZE * SIZE * INPUT_TENSORS];
    int axis = 0;
    int32_t *p;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_CHECKED(sizeof(struct onnx_node_t));
    node->ninput = INPUT_TENSORS;
    node->inputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->ninput);

    for (int i = 0; i < INPUT_TENSORS; ++i) {
        node->inputs[i] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
        node->inputs[i]->ndata = SIZE * SIZE;
        node->inputs[i]->ndim = 2;
        node->inputs[i]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[i]->ndim);
        node->inputs[i]->dims[0] = SIZE;
        node->inputs[i]->dims[1] = SIZE;
        node->inputs[i]->datas = MALLOC_CHECKED(sizeof(int32_t) * node->inputs[i]->ndata);
        p = (int32_t *)node->inputs[i]->datas;
        for (int j = 0; j < node->inputs[0]->ndata; j++) {
            p[j] = rand();
        }
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = SIZE * SIZE * INPUT_TENSORS;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE;
    node->outputs[0]->dims[1] = SIZE * INPUT_TENSORS;
    node->outputs[0]->datas = MALLOC_CHECKED(sizeof(int32_t) * node->outputs[0]->ndata);
    node->priv = &axis;

    // golden test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(Concat_int32_axis0);
    Concat_int32(node);
    BENCH_END(Concat_int32_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    // rvv optimization test with axis = 0
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int32_t));
    BENCH_START(Concat_int32_rvv_axis0);
    Concat_int32_rvv(node);
    BENCH_END(Concat_int32_rvv_axis0);

    // verify result with axis = 0
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with axis = 1
    axis = 1;
    node->outputs[0]->dims[0] = SIZE * INPUT_TENSORS;
    node->outputs[0]->dims[1] = SIZE;
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int32_t));
    BENCH_START(Concat_int32_axis1);
    Concat_int32(node);
    BENCH_END(Concat_int32_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    // rvv optimization test with axis = 1
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int32_t));
    BENCH_START(Concat_int32_rvv_axis1);
    Concat_int32_rvv(node);
    BENCH_END(Concat_int32_rvv_axis1);

    // verify result with axis = 1
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    for (int i = 0; i < node->ninput; ++i) {
        free(node->inputs[i]->datas);
        free(node->inputs[i]->dims);
        free(node->inputs[i]);
    }
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_concat_float16()
{
    struct onnx_node_t *node;
    float16_t golden[SIZE * SIZE * INPUT_TENSORS];
    int axis = 0;
    float16_t *p;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_CHECKED(sizeof(struct onnx_node_t));
    node->ninput = INPUT_TENSORS;
    node->inputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->ninput);

    for (int i = 0; i < INPUT_TENSORS; ++i) {
        node->inputs[i] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
        node->inputs[i]->ndata = SIZE * SIZE;
        node->inputs[i]->ndim = 2;
        node->inputs[i]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[i]->ndim);
        node->inputs[i]->dims[0] = SIZE;
        node->inputs[i]->dims[1] = SIZE;
        node->inputs[i]->datas = MALLOC_CHECKED(sizeof(float16_t) * node->inputs[i]->ndata);
        p = (float16_t *)node->inputs[i]->datas;
        for (int j = 0; j < node->inputs[0]->ndata; j++) {
            p[j] = rand() * 1.0f;
        }
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = SIZE * SIZE * INPUT_TENSORS;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE;
    node->outputs[0]->dims[1] = SIZE * INPUT_TENSORS;
    node->outputs[0]->datas = MALLOC_CHECKED(sizeof(float16_t) * node->outputs[0]->ndata);
    node->priv = &axis;

    // golden test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(Concat_float16_axis0);
    Concat_float16(node);
    BENCH_END(Concat_float16_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    // rvv optimization test with axis = 0
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(Concat_float16_rvv_axis0);
    Concat_float16_rvv(node);
    BENCH_END(Concat_float16_rvv_axis0);

    // verify result with axis = 0
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with axis = 1
    axis = 1;
    node->outputs[0]->dims[0] = SIZE * INPUT_TENSORS;
    node->outputs[0]->dims[1] = SIZE;
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(Concat_float16_axis1);
    Concat_float16(node);
    BENCH_END(Concat_float16_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    // rvv optimization test with axis = 1
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(Concat_float16_rvv_axis1);
    Concat_float16_rvv(node);
    BENCH_END(Concat_float16_rvv_axis1);

    // verify result with axis = 1
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    for (int i = 0; i < node->ninput; ++i) {
        free(node->inputs[i]->datas);
        free(node->inputs[i]->dims);
        free(node->inputs[i]);
    }
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_concat_float32()
{
    struct onnx_node_t *node;
    float32_t golden[SIZE * SIZE * INPUT_TENSORS];
    int axis = 0;
    float32_t *p;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_CHECKED(sizeof(struct onnx_node_t));
    node->ninput = INPUT_TENSORS;
    node->inputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->ninput);

    for (int i = 0; i < INPUT_TENSORS; ++i) {
        node->inputs[i] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
        node->inputs[i]->ndata = SIZE * SIZE;
        node->inputs[i]->ndim = 2;
        node->inputs[i]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->inputs[i]->ndim);
        node->inputs[i]->dims[0] = SIZE;
        node->inputs[i]->dims[1] = SIZE;
        node->inputs[i]->datas = MALLOC_CHECKED(sizeof(float32_t) * node->inputs[i]->ndata);
        p = (float32_t *)node->inputs[i]->datas;
        for (int j = 0; j < node->inputs[0]->ndata; j++) {
            p[j] = rand() * 1.0f;
        }
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_CHECKED(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_CHECKED(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = SIZE * SIZE * INPUT_TENSORS;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_CHECKED(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE;
    node->outputs[0]->dims[1] = SIZE * INPUT_TENSORS;
    node->outputs[0]->datas = MALLOC_CHECKED(sizeof(float32_t) * node->outputs[0]->ndata);
    node->priv = &axis;

    // golden test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(Concat_float32_axis0);
    Concat_float32(node);
    BENCH_END(Concat_float32_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    // rvv optimization test with axis = 0
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(Concat_float32_rvv_axis0);
    Concat_float32_rvv(node);
    BENCH_END(Concat_float32_rvv_axis0);

    // verify result with axis = 0
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with axis = 1
    axis = 1;
    node->outputs[0]->dims[0] = SIZE * INPUT_TENSORS;
    node->outputs[0]->dims[1] = SIZE;
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(Concat_float32_axis1);
    Concat_float32(node);
    BENCH_END(Concat_float32_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    // rvv optimization test with axis = 1
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(Concat_float32_rvv_axis1);
    Concat_float32_rvv(node);
    BENCH_END(Concat_float32_rvv_axis1);

    // verify result with axis = 1
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    for (int i = 0; i < node->ninput; ++i) {
        free(node->inputs[i]->datas);
        free(node->inputs[i]->dims);
        free(node->inputs[i]);
    }
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_concat(void)
{
    int ret = 0;
    ret |= test_concat_int8();
    ret |= test_concat_int32();
    ret |= test_concat_float16();
    ret |= test_concat_float32();
    return ret;
}