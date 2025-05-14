#include "utils.h"

BENCH_DECLARE_VAR()

#define SRC_WIDTH 128
#define SRC_HEIGHT 128

#define TARGET_WIDTH 192
#define TARGET_HEIGHT 192

int test_bilinear_interpolation_f16(void)
{
    struct onnx_node_t *node;
    uint8_t *golden;
    uint8_t *opt;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 1;

    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));

    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SRC_WIDTH;
    node->inputs[0]->dims[1] = SRC_HEIGHT;
    node->inputs[0]->ndata = node->inputs[0]->dims[0] * node->inputs[0]->dims[1];
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(uint8_t) * node->inputs[0]->ndata);

    uint8_t *p = (uint8_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = (uint8_t)(rand() % 256);
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));

    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = TARGET_WIDTH;
    node->outputs[0]->dims[1] = TARGET_HEIGHT;
    node->outputs[0]->ndata = node->outputs[0]->dims[0] * node->outputs[0]->dims[1];
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(uint8_t) * node->outputs[0]->ndata);

    golden = (uint8_t *)MALLOC_ASSERT(sizeof(uint8_t) * node->outputs[0]->ndata);
    opt = (uint8_t *)MALLOC_ASSERT(sizeof(uint8_t) * node->outputs[0]->ndata);

    BENCH_START(BilinearInterpolation_float16);
    BilinearInterpolation_float16(node);
    BENCH_END(BilinearInterpolation_float16);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(uint8_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(uint8_t));
    BENCH_START(BilinearInterpolation_float16_rvv);
    BilinearInterpolation_float16_rvv(node);
    BENCH_END(BilinearInterpolation_float16_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(uint8_t));

    ret |= verify_results_uint8(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    free(golden);
    free(opt);

    return ret;
}

int test_bilinear_interpolation_f32(void)
{
    struct onnx_node_t *node;
    uint8_t *golden;
    uint8_t *opt;
    int ret = 0;

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->ninput = 1;

    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));

    node->inputs[0]->ndim = 2;
    node->inputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->inputs[0]->ndim);
    node->inputs[0]->dims[0] = SRC_WIDTH;
    node->inputs[0]->dims[1] = SRC_HEIGHT;
    node->inputs[0]->ndata = node->inputs[0]->dims[0] * node->inputs[0]->dims[1];
    node->inputs[0]->datas = MALLOC_ASSERT(sizeof(uint8_t) * node->inputs[0]->ndata);

    uint8_t *p = (uint8_t *)node->inputs[0]->datas;
    for (int i = 0; i < node->inputs[0]->ndata; i++) {
        p[i] = (uint8_t)(rand() % 256);
    }

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));

    node->outputs[0]->ndim = 2;
    node->outputs[0]->dims = (int *)MALLOC_ASSERT(sizeof(int) * node->outputs[0]->ndim);
    node->outputs[0]->dims[0] = TARGET_WIDTH;
    node->outputs[0]->dims[1] = TARGET_HEIGHT;
    node->outputs[0]->ndata = node->outputs[0]->dims[0] * node->outputs[0]->dims[1];
    node->outputs[0]->datas = MALLOC_ASSERT(sizeof(uint8_t) * node->outputs[0]->ndata);

    golden = (uint8_t *)MALLOC_ASSERT(sizeof(uint8_t) * node->outputs[0]->ndata);
    opt = (uint8_t *)MALLOC_ASSERT(sizeof(uint8_t) * node->outputs[0]->ndata);

    BENCH_START(BilinearInterpolation_float32);
    BilinearInterpolation_float32(node);
    BENCH_END(BilinearInterpolation_float32);

    memcpy(golden, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(uint8_t));

    memset(node->outputs[0]->datas, 0, node->outputs[0]->ndata * sizeof(uint8_t));
    BENCH_START(BilinearInterpolation_float32_rvv);
    BilinearInterpolation_float32_rvv(node);
    BENCH_END(BilinearInterpolation_float32_rvv);
    memcpy(opt, node->outputs[0]->datas, node->outputs[0]->ndata * sizeof(uint8_t));

    ret |= verify_results_uint8(golden, opt, node->outputs[0]->ndata);

    free(node->inputs[0]->datas);
    free(node->inputs[0]->dims);
    free(node->inputs[0]);
    free(node->outputs[0]->datas);
    free(node->outputs[0]->dims);
    free(node->outputs[0]);
    free(node->inputs);
    free(node->outputs);
    free(node);

    free(golden);
    free(opt);

    return ret;
}

int test_bilinear_interpolation(void)
{
    int ret = 0;
    ret |= test_bilinear_interpolation_f32();
    ret |= test_bilinear_interpolation_f16();
    return ret;
}
