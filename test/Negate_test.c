#include "utils.h"

#define TEST_DATA_LEN 4096

BENCH_DECLARE_VAR()
int test_negate_int8(void)
{
    struct onnx_node_t *node;
    int8_t golden[TEST_DATA_LEN];
    int8_t opt[TEST_DATA_LEN];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 1;

    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = TEST_DATA_LEN;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(int8_t) * node->inputs[0]->ndata);

    int8_t *p = (int8_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() % 256 - 128;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = TEST_DATA_LEN;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(int8_t) * node->outputs[0]->ndata);

    BENCH_START(Negate_int8);
    Negate_int8(node);
    BENCH_END(Negate_int8);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int8_t));
    BENCH_START(Negate_int8_rvv);
    Negate_int8_rvv(node);
    BENCH_END(Negate_int8_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int8_t));

    ret |= verify_results_int8(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_negate_int32(void)
{
    struct onnx_node_t *node;
    int32_t golden[TEST_DATA_LEN];
    int32_t opt[TEST_DATA_LEN];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 1;

    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = TEST_DATA_LEN;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(int32_t) * node->inputs[0]->ndata);

    int32_t *p = (int32_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() - RAND_MAX / 2;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = TEST_DATA_LEN;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(int32_t) * node->outputs[0]->ndata);

    BENCH_START(Negate_int32);
    Negate_int32(node);
    BENCH_END(Negate_int32);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int32_t));
    BENCH_START(Negate_int32_rvv);
    Negate_int32_rvv(node);
    BENCH_END(Negate_int32_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    ret |= verify_results_int32(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    return ret;
}

int test_negate_f16(void)
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
        p[i] = (rand() - RAND_MAX / 2) * 1.0 / RAND_MAX * 10;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = TEST_DATA_LEN;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->outputs[0]->ndata);

    BENCH_START(Negate_float16);
    Negate_float16(node);
    BENCH_END(Negate_float16);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(Negate_float16_rvv);
    Negate_float16_rvv(node);
    BENCH_END(Negate_float16_rvv);
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

int test_negate_f32(void)
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
        p[i] = (rand() - RAND_MAX / 2) * 1.0 / RAND_MAX * 10;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = TEST_DATA_LEN;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->outputs[0]->ndata);

    BENCH_START(Negate_float32);
    Negate_float32(node);
    BENCH_END(Negate_float32);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(Negate_float32_rvv);
    Negate_float32_rvv(node);
    BENCH_END(Negate_float32_rvv);
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

int test_negate(void)
{
    int ret = 0;
    ret |= test_negate_int8();
    ret |= test_negate_int32();
    ret |= test_negate_f16();
    ret |= test_negate_f32();
    return ret;
}