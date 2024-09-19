#include "utils.h"

#define TEST_DATA_LEN 65536

BENCH_DECLARE_VAR()
int test_topk_int32(void)
{
    struct onnx_node_t *node;
    int32_t golden[TEST_DATA_LEN];
    int32_t opt[TEST_DATA_LEN];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 1;

    uint32_t k = 32;
#ifdef __riscv_vector
    k = __riscv_vlenb() / sizeof(int32_t) * 8;
#endif
    node->priv = GenerateTopkParam(k);

    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = TEST_DATA_LEN;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(int32_t) * node->inputs[0]->ndata);
    node->inputs[0]->ndim = 1;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = TEST_DATA_LEN;
    node->inputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;

    int32_t *p = (int32_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = rand() % 16384 - 8192;
    }

    // show_tensor_int32(node->inputs[0], "input");

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = k;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(int32_t) * node->outputs[0]->ndata);
    node->outputs[0]->ndim = 1;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = k;
    node->outputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;

    BENCH_START(Topk_int32);
    Topk_int32(node);
    BENCH_END(Topk_int32);
    HeapSort_int32((int32_t *)node->outputs[0]->datas, node->outputs[0]->ndata);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    // show_tensor_int32(node->outputs[0], "Topk_int32");

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(int32_t));
    BENCH_START(Topk_int32_rvv);
    Topk_int32_rvv(node);
    BENCH_END(Topk_int32_rvv);
    HeapSort_int32((int32_t *)node->outputs[0]->datas, node->outputs[0]->ndata);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(int32_t));

    // show_tensor_int32(node->outputs[0], "Topk_int32_rvv");

    ret |= verify_results_int32(golden, opt, node->outputs[0]->ndata);

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
    FreeTopkParam(&node->priv);
    free(node);

    return ret;
}

int test_topk_float16(void)
{
    struct onnx_node_t *node;
    float16_t golden[TEST_DATA_LEN];
    float16_t opt[TEST_DATA_LEN];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 1;
    uint32_t k = 64;
#ifdef __riscv_vector
    k = __riscv_vlenb() / sizeof(float16_t) * 8;
#endif
    node->priv = GenerateTopkParam(k);

    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = TEST_DATA_LEN;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->inputs[0]->ndata);
    node->inputs[0]->ndim = 1;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = TEST_DATA_LEN;
    node->inputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;

    float16_t *p = (float16_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = (rand() % 65536 - 32768) * 1.0 / 1024;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = k;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float16_t) * node->outputs[0]->ndata);
    node->outputs[0]->ndim = 1;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = k;
    node->outputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;

    BENCH_START(Topk_float16);
    Topk_float16(node);
    BENCH_END(Topk_float16);
    HeapSort_f16((float16_t *)node->outputs[0]->datas, node->outputs[0]->ndata);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float16_t));
    BENCH_START(Topk_float16_rvv);
    Topk_float16_rvv(node);
    BENCH_END(Topk_float16_rvv);
    HeapSort_f16((float16_t *)node->outputs[0]->datas, node->outputs[0]->ndata);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float16_t));

    ret |= verify_results_f16(golden, opt, node->outputs[0]->ndata);

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
    FreeTopkParam(&node->priv);
    free(node);

    return ret;
}

int test_topk_float32(void)
{
    struct onnx_node_t *node;
    float32_t golden[TEST_DATA_LEN];
    float32_t opt[TEST_DATA_LEN];
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 1;
    uint32_t k = 32;
#ifdef __riscv_vector
    k = __riscv_vlenb() / sizeof(float32_t) * 8;
#endif
    node->priv = GenerateTopkParam(k);

    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->inputs[0]->ndata = TEST_DATA_LEN;
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->inputs[0]->ndata);
    node->inputs[0]->ndim = 1;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = TEST_DATA_LEN;
    node->inputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->strides[0] = 1;

    float32_t *p = (float32_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = (rand() % 65536 - 32768) * 1.0 / 1024;
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    node->outputs[0]->ndata = k;
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(float32_t) * node->outputs[0]->ndata);
    node->outputs[0]->ndim = 1;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = k;
    node->outputs[0]->strides = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->strides[0] = 1;

    BENCH_START(Topk_float32);
    Topk_float32(node);
    BENCH_END(Topk_float32);
    HeapSort_f32((float32_t *)node->outputs[0]->datas, node->outputs[0]->ndata);
    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(float32_t));
    BENCH_START(Topk_float32_rvv);
    Topk_float32_rvv(node);
    BENCH_END(Topk_float32_rvv);
    HeapSort_f32((float32_t *)node->outputs[0]->datas, node->outputs[0]->ndata);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(float32_t));

    ret |= verify_results_f32(golden, opt, node->outputs[0]->ndata);

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
    FreeTopkParam(&node->priv);
    free(node);

    return ret;
}

int test_topk(void)
{
    int ret = 0;
    ret |= test_topk_int32();
    ret |= test_topk_float16();
    ret |= test_topk_float32();
    return ret;
}