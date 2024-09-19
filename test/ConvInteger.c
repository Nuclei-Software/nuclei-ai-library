#include "utils.h"

BENCH_DECLARE_VAR()

#define CIN (96)
#define COUT (96)
#define IN_SZ (8)
#define OUT_SZ (8)

int test_convinteger(void)
{
    struct onnx_node_t *node;
    int ret = 0;

    struct onnx_tensor_t *input = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    input->ndim = 4;
    input->dims = (int *)MALLOC_ASSERT(sizeof(int) * input->ndim);
    input->dims[0] = CIN;
    input->dims[1] = IN_SZ;
    input->dims[2] = IN_SZ;
    input->dims[3] = 1;
    input->type = ONNX_TENSOR_TYPE_INT8;
    input->ndata = input->dims[0] * input->dims[1] * input->dims[2] * input->dims[3];
    input->datas = MALLOC_ASSERT(sizeof(int8_t) * input->ndata);
    for (int i = 0; i < input->ndata; i++) {
        ((int8_t *)input->datas)[i] = rand() % 256 - 128;
    }

    struct onnx_tensor_t *filter = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    filter->ndim = 4;
    filter->dims = (int *)MALLOC_ASSERT(sizeof(int) * filter->ndim);
    filter->dims[0] = CIN;
    filter->dims[1] = 3;
    filter->dims[2] = 3;
    filter->dims[3] = COUT;
    filter->type = ONNX_TENSOR_TYPE_INT8;
    filter->ndata = filter->dims[0] * filter->dims[1] * filter->dims[2] * filter->dims[3];
    filter->datas = MALLOC_ASSERT(sizeof(int8_t) * filter->ndata);
    for (int i = 0; i < filter->ndata; i++) {
        ((int8_t *)filter->datas)[i] = rand() % 256 - 128;
    }

    struct onnx_tensor_t *output_ref = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    output_ref->ndim = 4;
    output_ref->dims = (int *)MALLOC_ASSERT(sizeof(int) * output_ref->ndim);
    output_ref->dims[0] = COUT;
    output_ref->dims[1] = OUT_SZ;
    output_ref->dims[2] = OUT_SZ;
    output_ref->dims[3] = 1;
    output_ref->type = ONNX_TENSOR_TYPE_INT8;
    output_ref->ndata = output_ref->dims[0] * output_ref->dims[1] * output_ref->dims[2] * output_ref->dims[3];
    output_ref->datas = MALLOC_ASSERT(sizeof(int8_t) * output_ref->ndata);
    memset(output_ref->datas, 0, sizeof(int8_t) * output_ref->ndata);

    struct onnx_tensor_t *bias = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    bias->ndim = 1;
    bias->dims = (int *)MALLOC_ASSERT(sizeof(int) * bias->ndim);
    bias->dims[0] = COUT;
    bias->ndata = bias->dims[0];
    bias->type = ONNX_TENSOR_TYPE_INT32;
    bias->datas = MALLOC_ASSERT(sizeof(int32_t) * bias->ndata);
    memset(bias->datas, 0, sizeof(int32_t) * bias->ndata);

    struct onnx_tensor_t *multiply = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    multiply->ndim = 1;
    multiply->dims = (int *)MALLOC_ASSERT(sizeof(int) * multiply->ndim);
    multiply->dims[0] = COUT;
    multiply->ndata = multiply->dims[0];
    multiply->type = ONNX_TENSOR_TYPE_INT32;
    multiply->datas = MALLOC_ASSERT(sizeof(int32_t) * multiply->ndata);
    for (int i = 0; i < multiply->ndata; i++) {
        ((int32_t *)multiply->datas)[i] = 0x1000000;
    }

    struct onnx_tensor_t *shift = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    shift->ndim = 1;
    shift->dims = (int *)MALLOC_ASSERT(sizeof(int) * shift->ndim);
    shift->dims[0] = COUT;
    shift->ndata = shift->dims[0];
    shift->type = ONNX_TENSOR_TYPE_INT32;
    shift->datas = MALLOC_ASSERT(sizeof(int32_t) * shift->ndata);
    for (int i = 0; i < shift->ndata; i++) {
        ((int32_t *)shift->datas)[i] = -2;
    }

    node = (struct onnx_node_t *)MALLOC_ASSERT(sizeof(struct onnx_node_t));
    node->priv = GenerateConvIntegerParam(0, 0, 1, 1, 1, 1, 1, 1, -128, 127, input, filter, output_ref, 0);

    node->ninput = 5;
    node->inputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->ninput);
    node->inputs[0] = input;
    node->inputs[1] = filter;
    node->inputs[2] = bias;
    node->inputs[3] = multiply;
    node->inputs[4] = shift;

    node->noutput = 1;
    node->outputs = (struct onnx_tensor_t **)MALLOC_ASSERT(sizeof(struct onnx_tensor_t *) * node->noutput);
    node->outputs[0] = output_ref;

    BENCH_START(ConvInteger_Int8);
    ConvInteger(node);
    BENCH_END(ConvInteger_Int8);
    FreeConvIntegerParam(&node->priv);

    // allocate new buffer for rvv test
    struct onnx_tensor_t *output_rvv = (struct onnx_tensor_t *)MALLOC_ASSERT(sizeof(struct onnx_tensor_t));
    output_rvv->ndim = 4;
    output_rvv->dims = (int *)MALLOC_ASSERT(sizeof(int) * output_rvv->ndim);
    output_rvv->dims[0] = COUT;
    output_rvv->dims[1] = OUT_SZ;
    output_rvv->dims[2] = OUT_SZ;
    output_rvv->dims[3] = 1;
    output_rvv->type = ONNX_TENSOR_TYPE_INT8;
    output_rvv->ndata = output_rvv->dims[0] * output_rvv->dims[1] * output_rvv->dims[2] * output_rvv->dims[3];
    output_rvv->datas = MALLOC_ASSERT(sizeof(int8_t) * output_rvv->ndata);
    memset(output_rvv->datas, 0, sizeof(int8_t) * output_rvv->ndata);
    node->outputs[0] = output_rvv;
    node->priv = GenerateConvIntegerParam(0, 0, 1, 1, 1, 1, 1, 1, -128, 127, input, filter, output_rvv, 1);

    // run rvv test
    BENCH_START(ConvInteger_Int8_rvv);
    ConvInteger_rvv(node);
    BENCH_END(ConvInteger_Int8_rvv);
    FreeConvIntegerParam(&node->priv);

    // verify result
    ret |= verify_results_int8(output_ref->datas, output_rvv->datas, node->outputs[0]->ndata);

    free(input->datas);
    free(filter->datas);
    free(bias->datas);
    free(multiply->datas);
    free(shift->datas);
    free(input->dims);
    free(filter->dims);
    free(bias->dims);
    free(multiply->dims);
    free(shift->dims);
    free(input);
    free(filter);
    free(bias);
    free(multiply);
    free(shift);
    free(node->inputs);
    free(output_ref->datas);
    free(output_ref->dims);
    free(output_rvv->datas);
    free(output_rvv->dims);
    free(output_ref);
    free(output_rvv);
    free(node->outputs);
    free(node);

    return ret;
}