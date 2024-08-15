#include "utils.h"

#define SIZE 32

#define PAD_TOP 1
#define PAD_BOTTOM 2
#define PAD_LEFT 3
#define PAD_RIGHT 4

BENCH_DECLARE_VAR()

int test_pad_int8()
{
    struct onnx_node_t *node;
    int8_t golden[(SIZE + PAD_TOP + PAD_BOTTOM) * (SIZE + PAD_LEFT + PAD_RIGHT)], *p;
    const OnnxScalar pad_const = {.v_uint8 = 0x55};
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = SIZE * SIZE;
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SIZE;
    node->inputs[0]->dims[1] = SIZE;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(int8_t) * node->inputs[0]->ndata);
    p = (int8_t *)node->inputs[0]->datas;
    for (int i = 0; i < SIZE * SIZE; i++) {
        p[i] = rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = (SIZE + PAD_TOP + PAD_BOTTOM) * (SIZE + PAD_LEFT + PAD_RIGHT);
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE + PAD_LEFT + PAD_RIGHT;
    node->outputs[0]->dims[1] = SIZE + PAD_TOP + PAD_BOTTOM;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(int8_t) * node->outputs[0]->ndata);

    node->priv = GeneratePadParam(pad_const, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT);

    // golden test
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(Pad_int8);
    Pad_int8(node);
    BENCH_END(Pad_int8);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    // rvv optimization test
    memset(node->outputs[0]->datas, 0, sizeof(int8_t) * node->outputs[0]->ndata);
    BENCH_START(Pad_int8_rvv);
    Pad_int8_rvv(node);
    BENCH_END(Pad_int8_rvv);

    // verify results
    ret |= verify_results_int8(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    FreePadParam(&node->priv);
    free(node);

    return ret;
}

int test_pad_int32()
{
    struct onnx_node_t *node;
    int32_t golden[(SIZE + PAD_TOP + PAD_BOTTOM) * (SIZE + PAD_LEFT + PAD_RIGHT)], *p;
    const OnnxScalar pad_const = {.v_int32 = 0x55555555};
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = SIZE * SIZE;
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SIZE;
    node->inputs[0]->dims[1] = SIZE;
    node->inputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = SIZE;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(int32_t) * node->inputs[0]->ndata);
    p = (int32_t *)node->inputs[0]->datas;
    for (int i = 0; i < SIZE * SIZE; i++) {
        p[i] = rand();
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = (SIZE + PAD_TOP + PAD_BOTTOM) * (SIZE + PAD_LEFT + PAD_RIGHT);
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE + PAD_LEFT + PAD_RIGHT;
    node->outputs[0]->dims[1] = SIZE + PAD_TOP + PAD_BOTTOM;
    node->outputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = SIZE + PAD_LEFT + PAD_RIGHT;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(int32_t) * node->outputs[0]->ndata);

    node->priv = GeneratePadParam(pad_const, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT);

    // golden test
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(Pad_int32);
    Pad_int32(node);
    BENCH_END(Pad_int32);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    // rvv optimization test
    memset(node->outputs[0]->datas, 0, sizeof(int32_t) * node->outputs[0]->ndata);
    BENCH_START(Pad_int32_rvv);
    Pad_int32_rvv(node);
    BENCH_END(Pad_int32_rvv);

    // verify results
    ret |= verify_results_int32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]->strides);
    free(node->inputs[0]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]->strides);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    FreePadParam(&node->priv);
    free(node);

    return ret;
}

int test_pad_float16()
{
    struct onnx_node_t *node;
    float16_t golden[(SIZE + PAD_TOP + PAD_BOTTOM) * (SIZE + PAD_LEFT + PAD_RIGHT)], *p;
    const OnnxScalar pad_const = {.v_float16 = 0.5f};
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = SIZE * SIZE;
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SIZE;
    node->inputs[0]->dims[1] = SIZE;
    node->inputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = SIZE;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->inputs[0]->ndata);
    p = (float16_t *)node->inputs[0]->datas;
    for (int i = 0; i < SIZE * SIZE; i++) {
        p[i] = rand() * 1.0f;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = (SIZE + PAD_TOP + PAD_BOTTOM) * (SIZE + PAD_LEFT + PAD_RIGHT);
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE + PAD_LEFT + PAD_RIGHT;
    node->outputs[0]->dims[1] = SIZE + PAD_TOP + PAD_BOTTOM;
    node->outputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = SIZE + PAD_LEFT + PAD_RIGHT;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->outputs[0]->ndata);

    node->priv = GeneratePadParam(pad_const, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT);

    // golden test
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(Pad_float16);
    Pad_float16(node);
    BENCH_END(Pad_float16);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    // rvv optimization test
    memset(node->outputs[0]->datas, 0, sizeof(float16_t) * node->outputs[0]->ndata);
    BENCH_START(Pad_float16_rvv);
    Pad_float16_rvv(node);
    BENCH_END(Pad_float16_rvv);

    // verify results
    ret |= verify_results_f16(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]->strides);
    free(node->inputs[0]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]->strides);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    FreePadParam(&node->priv);
    free(node);

    return ret;
}

int test_pad_float32()
{
    struct onnx_node_t *node;
    float32_t golden[(SIZE + PAD_TOP + PAD_BOTTOM) * (SIZE + PAD_LEFT + PAD_RIGHT)], *p;
    const OnnxScalar pad_const = {.v_float32 = 0.5f};
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 1;
    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);

    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = SIZE * SIZE;
    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SIZE;
    node->inputs[0]->dims[1] = SIZE;
    node->inputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;
    node->inputs[0]->strides[1] = SIZE;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->inputs[0]->ndata);
    p = (float32_t *)node->inputs[0]->datas;
    for (int i = 0; i < SIZE * SIZE; i++) {
        p[i] = rand() * 1.0f;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = (SIZE + PAD_TOP + PAD_BOTTOM) * (SIZE + PAD_LEFT + PAD_RIGHT);
    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = SIZE + PAD_LEFT + PAD_RIGHT;
    node->outputs[0]->dims[1] = SIZE + PAD_TOP + PAD_BOTTOM;
    node->outputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;
    node->outputs[0]->strides[1] = SIZE + PAD_LEFT + PAD_RIGHT;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->outputs[0]->ndata);

    node->priv = GeneratePadParam(pad_const, PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT);

    // golden test
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(Pad_float32);
    Pad_float32(node);
    BENCH_END(Pad_float32);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    // rvv optimization test
    memset(node->outputs[0]->datas, 0, sizeof(float32_t) * node->outputs[0]->ndata);
    BENCH_START(Pad_float32_rvv);
    Pad_float32_rvv(node);
    BENCH_END(Pad_float32_rvv);

    // verify results
    ret |= verify_results_f32(golden, node->outputs[0]->datas, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]->strides);
    free(node->inputs[0]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]->strides);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    FreePadParam(&node->priv);
    free(node);

    return ret;
}

int test_pad(void)
{
    int ret = 0;
    ret |= test_pad_int8();
    ret |= test_pad_int32();
    ret |= test_pad_float16();
    ret |= test_pad_float32();
    return ret;
}