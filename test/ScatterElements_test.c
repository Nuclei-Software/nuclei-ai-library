#include "utils.h"

#define SIZE 64
#define SRC_SIZE 32

BENCH_DECLARE_VAR()

// NOTE: the indices using uint8_t to store, so the max SIZE is 256

int test_scatterelements_int8()
{
    struct onnx_node_t *node;
    int8_t golden[SIZE * SIZE];
    int8_t *p;
    uint8_t *pidx;
    int axis = 0;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 2;
    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);

    // src tensor
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = SRC_SIZE * SRC_SIZE;
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SRC_SIZE;
    node->inputs[0]->dims[1] = SRC_SIZE;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(int8_t) * node->inputs[0]->ndata);
    p = (int8_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    // indices tensor
    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = SRC_SIZE * SRC_SIZE;
    node->inputs[1]->ndim = 2;
    node->inputs[1]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[1]->ndim);
    node->inputs[1]->dims[0] = SRC_SIZE;
    node->inputs[1]->dims[1] = SRC_SIZE;
    node->inputs[1]->datas = MALLOC_ASSERT(sizeof(uint8_t) * node->inputs[1]->ndata);
    pidx = (uint8_t *)node->inputs[1]->datas;
    for (int i = 0; i < SRC_SIZE; ++i) {
        for (int j = 0; j < SRC_SIZE; ++j) {
            // when axis == 0, index is scatter along axis 0
            pidx[i * SRC_SIZE + j] = (i * 2) % SIZE;
        }
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = SIZE * SIZE;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE;
    node->outputs[0]->dims[1] = SIZE;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(int8_t) * node->outputs[0]->ndata);
    node->priv = &axis;

    // golden test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_int8_axis0);
    ScatterElements_int8(node);
    BENCH_END(ScatterElements_int8_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_int8_rvv_axis0);
    ScatterElements_int8_rvv(node);
    BENCH_END(ScatterElements_int8_rvv_axis0);

    // verify result with axis = 0
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with axis = 1
    axis = 1;
    for (int i = 0; i < SRC_SIZE; ++i) {
        for (int j = 0; j < SRC_SIZE; ++j) {
            // when axis == 0, index is scatter along axis 0
            pidx[i * SRC_SIZE + j] = (j * 2) % SIZE;
        }
    }
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_int8_axis1);
    ScatterElements_int8(node);
    BENCH_END(ScatterElements_int8_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test with axis = 1
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_int8_rvv_axis1);
    ScatterElements_int8_rvv(node);
    BENCH_END(ScatterElements_int8_rvv_axis1);

    // verify result with axis = 1
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

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

int test_scatterelements_int32()
{
    struct onnx_node_t *node;
    int32_t golden[SIZE * SIZE];
    int32_t *p;
    uint8_t *pidx;
    int axis = 0;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 2;
    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);

    // src tensor
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = SRC_SIZE * SRC_SIZE;
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SRC_SIZE;
    node->inputs[0]->dims[1] = SRC_SIZE;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(int32_t) * node->inputs[0]->ndata);
    p = (int32_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    // indices tensor
    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = SRC_SIZE * SRC_SIZE;
    node->inputs[1]->ndim = 2;
    node->inputs[1]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[1]->ndim);
    node->inputs[1]->dims[0] = SRC_SIZE;
    node->inputs[1]->dims[1] = SRC_SIZE;
    node->inputs[1]->datas = MALLOC_ASSERT(sizeof(uint8_t) * node->inputs[1]->ndata);
    pidx = (uint8_t *)node->inputs[1]->datas;
    for (int i = 0; i < SRC_SIZE; ++i) {
        for (int j = 0; j < SRC_SIZE; ++j) {
            // when axis == 0, index is scatter along axis 0
            pidx[i * SRC_SIZE + j] = (i * 2) % SIZE;
        }
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = SIZE * SIZE;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE;
    node->outputs[0]->dims[1] = SIZE;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(int32_t) * node->outputs[0]->ndata);
    node->priv = &axis;

    // golden test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_int32_axis0);
    ScatterElements_int32(node);
    BENCH_END(ScatterElements_int32_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    // rvv optimization test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_int32_rvv_axis0);
    ScatterElements_int32_rvv(node);
    BENCH_END(ScatterElements_int32_rvv_axis0);

    // verify result with axis = 0
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with axis = 1
    axis = 1;
    for (int i = 0; i < SRC_SIZE; ++i) {
        for (int j = 0; j < SRC_SIZE; ++j) {
            // when axis == 0, index is scatter along axis 0
            pidx[i * SRC_SIZE + j] = (j * 2) % SIZE;
        }
    }
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_int32_axis1);
    ScatterElements_int32(node);
    BENCH_END(ScatterElements_int32_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    // rvv optimization test with axis = 1
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_int32_rvv_axis1);
    ScatterElements_int32_rvv(node);
    BENCH_END(ScatterElements_int32_rvv_axis1);

    // verify result with axis = 1
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

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

int test_scatterelements_float16()
{
    struct onnx_node_t *node;
    float16_t golden[SIZE * SIZE];
    float16_t *p;
    uint8_t *pidx;
    int axis = 0;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 2;
    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);

    // src tensor
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = SRC_SIZE * SRC_SIZE;
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SRC_SIZE;
    node->inputs[0]->dims[1] = SRC_SIZE;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->inputs[0]->ndata);
    p = (float16_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    // indices tensor
    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = SRC_SIZE * SRC_SIZE;
    node->inputs[1]->ndim = 2;
    node->inputs[1]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[1]->ndim);
    node->inputs[1]->dims[0] = SRC_SIZE;
    node->inputs[1]->dims[1] = SRC_SIZE;
    node->inputs[1]->datas = MALLOC_ASSERT(sizeof(uint8_t) * node->inputs[1]->ndata);
    pidx = (uint8_t *)node->inputs[1]->datas;
    for (int i = 0; i < SRC_SIZE; ++i) {
        for (int j = 0; j < SRC_SIZE; ++j) {
            // when axis == 0, index is scatter along axis 0
            pidx[i * SRC_SIZE + j] = (i * 2) % SIZE;
        }
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = SIZE * SIZE;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE;
    node->outputs[0]->dims[1] = SIZE;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->outputs[0]->ndata);
    node->priv = &axis;

    // golden test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_float16_axis0);
    ScatterElements_float16(node);
    BENCH_END(ScatterElements_float16_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    // rvv optimization test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_float16_rvv_axis0);
    ScatterElements_float16_rvv(node);
    BENCH_END(ScatterElements_float16_rvv_axis0);

    // verify result with axis = 0
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with axis = 1
    axis = 1;
    for (int i = 0; i < SRC_SIZE; ++i) {
        for (int j = 0; j < SRC_SIZE; ++j) {
            // when axis == 0, index is scatter along axis 0
            pidx[i * SRC_SIZE + j] = (j * 2) % SIZE;
        }
    }
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_float16_axis1);
    ScatterElements_float16(node);
    BENCH_END(ScatterElements_float16_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    // rvv optimization test with axis = 1
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_float16_rvv_axis1);
    ScatterElements_float16_rvv(node);
    BENCH_END(ScatterElements_float16_rvv_axis1);

    // verify result with axis = 1
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

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

int test_scatterelements_float32()
{
    struct onnx_node_t *node;
    float32_t golden[SIZE * SIZE];
    float32_t *p;
    uint8_t *pidx;
    int axis = 0;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 2;
    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);

    // src tensor
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = SRC_SIZE * SRC_SIZE;
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SRC_SIZE;
    node->inputs[0]->dims[1] = SRC_SIZE;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->inputs[0]->ndata);
    p = (float32_t *)node->inputs[0]->datas;
    for (int j = 0; j < node->inputs[0]->ndata; j++) {
        p[j] = rand();
    }

    // indices tensor
    node->inputs[1] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[1]->ndata = SRC_SIZE * SRC_SIZE;
    node->inputs[1]->ndim = 2;
    node->inputs[1]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[1]->ndim);
    node->inputs[1]->dims[0] = SRC_SIZE;
    node->inputs[1]->dims[1] = SRC_SIZE;
    node->inputs[1]->datas = MALLOC_ASSERT(sizeof(uint8_t) * node->inputs[1]->ndata);
    pidx = (uint8_t *)node->inputs[1]->datas;
    for (int i = 0; i < SRC_SIZE; ++i) {
        for (int j = 0; j < SRC_SIZE; ++j) {
            // when axis == 0, index is scatter along axis 0
            pidx[i * SRC_SIZE + j] = (i * 2) % SIZE;
        }
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = SIZE * SIZE;
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE;
    node->outputs[0]->dims[1] = SIZE;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->outputs[0]->ndata);
    node->priv = &axis;

    // golden test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_float32_axis0);
    ScatterElements_float32(node);
    BENCH_END(ScatterElements_float32_axis0);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    // rvv optimization test with axis = 0
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_float32_rvv_axis0);
    ScatterElements_float32_rvv(node);
    BENCH_END(ScatterElements_float32_rvv_axis0);

    // verify result with axis = 0
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    // golden test with axis = 1
    axis = 1;
    for (int i = 0; i < SRC_SIZE; ++i) {
        for (int j = 0; j < SRC_SIZE; ++j) {
            // when axis == 0, index is scatter along axis 0
            pidx[i * SRC_SIZE + j] = (j * 2) % SIZE;
        }
    }
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_float32_axis1);
    ScatterElements_float32(node);
    BENCH_END(ScatterElements_float32_axis1);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    // rvv optimization test with axis = 1
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(ScatterElements_float32_rvv_axis1);
    ScatterElements_float32_rvv(node);
    BENCH_END(ScatterElements_float32_rvv_axis1);

    // verify result with axis = 1
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

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

int test_scatterelements()
{
    int ret = 0;
    ret |= test_scatterelements_int8();
    ret |= test_scatterelements_int32();
    ret |= test_scatterelements_float16();
    ret |= test_scatterelements_float32();
    return ret;
}