/**
 * @file Tile_test.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-11
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "utils.h"

#define SIZE 32
#define TIMES 2

BENCH_DECLARE_VAR()

int test_tile_int8()
{
    struct onnx_node_t *node;
    int8_t golden[SIZE * SIZE * TIMES * TIMES];
    int8_t *p;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 2;
    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = SIZE * SIZE;
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SIZE;
    node->inputs[0]->dims[1] = SIZE;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(int8_t) * node->inputs[0]->ndata);
    p = (int8_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = 2;
    node->inputs[1]->ndim = 1;
    node->inputs[1]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[1]->ndim);
    node->inputs[1]->dims[0] = 2;
    node->inputs[1]->datas = MALLOC_ASSERT(sizeof(int) * node->inputs[1]->ndata);
    ((int *)(node->inputs[1]->datas))[0] = TIMES;
    ((int *)(node->inputs[1]->datas))[1] = 1;

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = SIZE * SIZE * TIMES * TIMES;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE;
    node->outputs[0]->dims[1] = SIZE;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(int8_t) * node->outputs[0]->ndata);

    // golden test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(Tile_int8_axis0);
    Tile_int8(node);
    BENCH_END(Tile_int8_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test with axis = 0
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int8_t));
    BENCH_START(Tile_int8_rvv_axis0);
    Tile_int8_rvv(node);
    BENCH_END(Tile_int8_rvv_axis0);

    // verify result with axis = 0
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with axis = 1
    ((int *)(node->inputs[1]->datas))[0] = 1;
    ((int *)(node->inputs[1]->datas))[1] = TIMES;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(Tile_int8_axis1);
    Tile_int8(node);
    BENCH_END(Tile_int8_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test with axis = 1
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int8_t));
    BENCH_START(Tile_int8_rvv_axis1);
    Tile_int8_rvv(node);
    BENCH_END(Tile_int8_rvv_axis1);

    // verify result with axis = 1
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with both axis
    ((int *)(node->inputs[1]->datas))[0] = TIMES;
    ((int *)(node->inputs[1]->datas))[1] = TIMES;
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(Tile_int8_bothaxes);
    Tile_int8(node);
    BENCH_END(Tile_int8_bothaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test with both axes
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int8_t));
    BENCH_START(Tile_int8_rvv_bothaxes);
    Tile_int8_rvv(node);
    BENCH_END(Tile_int8_rvv_bothaxes);

    // verify result with both axis
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs[1]->datas);
    free(node->inputs[1]->dims);
    free(node->inputs[1]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]->strides);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_tile_int32()
{
    struct onnx_node_t *node;
    int32_t golden[SIZE * SIZE * TIMES * TIMES];
    int32_t *p;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 2;
    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = SIZE * SIZE;
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SIZE;
    node->inputs[0]->dims[1] = SIZE;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(int32_t) * node->inputs[0]->ndata);
    p = (int32_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = 2;
    node->inputs[1]->ndim = 1;
    node->inputs[1]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[1]->ndim);
    node->inputs[1]->dims[0] = 2;
    node->inputs[1]->datas = MALLOC_ASSERT(sizeof(int) * node->inputs[1]->ndata);
    ((int *)(node->inputs[1]->datas))[0] = TIMES;
    ((int *)(node->inputs[1]->datas))[1] = 1;

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = SIZE * SIZE * TIMES * TIMES;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE;
    node->outputs[0]->dims[1] = SIZE;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(int32_t) * node->outputs[0]->ndata);

    // golden test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(Tile_int32_axis0);
    Tile_int32(node);
    BENCH_END(Tile_int32_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    // rvv optimization test with axis = 0
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int32_t));
    BENCH_START(Tile_int32_rvv_axis0);
    Tile_int32_rvv(node);
    BENCH_END(Tile_int32_rvv_axis0);

    // verify result with axis = 0
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with axis = 1
    ((int *)(node->inputs[1]->datas))[0] = 1;
    ((int *)(node->inputs[1]->datas))[1] = TIMES;
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(Tile_int32_axis1);
    Tile_int32(node);
    BENCH_END(Tile_int32_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    // rvv optimization test with axis = 1
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int32_t));
    BENCH_START(Tile_int32_rvv_axis1);
    Tile_int32_rvv(node);
    BENCH_END(Tile_int32_rvv_axis1);

    // verify result with axis = 1
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with both axis
    ((int *)(node->inputs[1]->datas))[0] = TIMES;
    ((int *)(node->inputs[1]->datas))[1] = TIMES;
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(Tile_int32_bothaxes);
    Tile_int32(node);
    BENCH_END(Tile_int32_bothaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    // rvv optimization test with both axes
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int32_t));
    BENCH_START(Tile_int32_rvv_bothaxes);
    Tile_int32_rvv(node);
    BENCH_END(Tile_int32_rvv_bothaxes);

    // verify result with both axis
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs[1]->datas);
    free(node->inputs[1]->dims);
    free(node->inputs[1]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]->strides);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_tile_float32()
{
    struct onnx_node_t *node;
    float32_t golden[SIZE * SIZE * TIMES * TIMES];
    float32_t *p;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 2;
    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = SIZE * SIZE;
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SIZE;
    node->inputs[0]->dims[1] = SIZE;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->inputs[0]->ndata);
    p = (float32_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = 2;
    node->inputs[1]->ndim = 1;
    node->inputs[1]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[1]->ndim);
    node->inputs[1]->dims[0] = 2;
    node->inputs[1]->datas = MALLOC_ASSERT(sizeof(int) * node->inputs[1]->ndata);
    ((int *)(node->inputs[1]->datas))[0] = TIMES;
    ((int *)(node->inputs[1]->datas))[1] = 1;

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = SIZE * SIZE * TIMES * TIMES;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE;
    node->outputs[0]->dims[1] = SIZE;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->outputs[0]->ndata);

    // golden test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(Tile_float32_axis0);
    Tile_float32(node);
    BENCH_END(Tile_float32_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    // rvv optimization test with axis = 0
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(Tile_float32_rvv_axis0);
    Tile_float32_rvv(node);
    BENCH_END(Tile_float32_rvv_axis0);

    // verify result with axis = 0
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with axis = 1
    ((int *)(node->inputs[1]->datas))[0] = 1;
    ((int *)(node->inputs[1]->datas))[1] = TIMES;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(Tile_float32_axis1);
    Tile_float32(node);
    BENCH_END(Tile_float32_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    // rvv optimization test with axis = 1
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(Tile_float32_rvv_axis1);
    Tile_float32_rvv(node);
    BENCH_END(Tile_float32_rvv_axis1);

    // verify result with axis = 1
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with both axis
    ((int *)(node->inputs[1]->datas))[0] = TIMES;
    ((int *)(node->inputs[1]->datas))[1] = TIMES;
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(Tile_float32_bothaxes);
    Tile_float32(node);
    BENCH_END(Tile_float32_bothaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    // rvv optimization test with both axes
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(Tile_float32_rvv_bothaxes);
    Tile_float32_rvv(node);
    BENCH_END(Tile_float32_rvv_bothaxes);

    // verify result with both axis
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs[1]->datas);
    free(node->inputs[1]->dims);
    free(node->inputs[1]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]->strides);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_tile_float16()
{
    struct onnx_node_t *node;
    float16_t golden[SIZE * SIZE * TIMES * TIMES];
    float16_t *p;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 2;
    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = SIZE * SIZE;
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SIZE;
    node->inputs[0]->dims[1] = SIZE;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->inputs[0]->ndata);
    p = (float16_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = 2;
    node->inputs[1]->ndim = 1;
    node->inputs[1]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[1]->ndim);
    node->inputs[1]->dims[0] = 2;
    node->inputs[1]->datas = MALLOC_ASSERT(sizeof(int) * node->inputs[1]->ndata);
    ((int *)(node->inputs[1]->datas))[0] = TIMES;
    ((int *)(node->inputs[1]->datas))[1] = 1;

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = SIZE * SIZE * TIMES * TIMES;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE;
    node->outputs[0]->dims[1] = SIZE;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->outputs[0]->ndata);

    // golden test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(Tile_float16_axis0);
    Tile_float16(node);
    BENCH_END(Tile_float16_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    // rvv optimization test with axis = 0
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(Tile_float16_rvv_axis0);
    Tile_float16_rvv(node);
    BENCH_END(Tile_float16_rvv_axis0);

    // verify result with axis = 0
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with axis = 1
    ((int *)(node->inputs[1]->datas))[0] = 1;
    ((int *)(node->inputs[1]->datas))[1] = TIMES;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(Tile_float16_axis1);
    Tile_float16(node);
    BENCH_END(Tile_float16_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    // rvv optimization test with axis = 1
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(Tile_float16_rvv_axis1);
    Tile_float16_rvv(node);
    BENCH_END(Tile_float16_rvv_axis1);

    // verify result with axis = 1
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with both axis
    ((int *)(node->inputs[1]->datas))[0] = TIMES;
    ((int *)(node->inputs[1]->datas))[1] = TIMES;
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(Tile_float16_bothaxes);
    Tile_float16(node);
    BENCH_END(Tile_float16_bothaxes);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    // rvv optimization test with both axes
    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(Tile_float16_rvv_bothaxes);
    Tile_float16_rvv(node);
    BENCH_END(Tile_float16_rvv_bothaxes);

    // verify result with both axis
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->inputs[1]->datas);
    free(node->inputs[1]->dims);
    free(node->inputs[1]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]->strides);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_tile(void)
{
    int ret = 0;
    ret |= test_tile_int8();
    ret |= test_tile_int32();
    ret |= test_tile_float16();
    ret |= test_tile_float32();
    return ret;
}