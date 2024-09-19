#ifndef __OPERATORS_H__
#define __OPERATORS_H__

#include "onnx.h"

typedef union onnx_scalar_t {
    uint8_t v_bool;
    int8_t v_int8;
    int16_t v_int16;
    int32_t v_int32;
    int64_t v_int64;
    uint8_t v_uint8;
    uint16_t v_uint16;
    uint32_t v_uint32;
    uint64_t v_uint64;
    uint16_t v_bfloat16;
    float16_t v_float16;
    float v_float32;
} OnnxScalar;

/* ---------------- start of helper function ----------------- */

void *GenerateBatchNormParam(float epsilon, float momentum);
void FreeBatchNormParam(void **pdat);
void *GenerateLayerNormParam(float epsilon, float momentum);
void FreeLayerNormParam(void **pdat);
void *GenerateRMSNormParam(float epsilon, float momentum);
void FreeRMSNormParam(void **pdat);
void *GenerateTopkParam(uint32_t k);
void FreeTopkParam(void **pdat);
void *GenerateClampParam(OnnxScalar min, OnnxScalar max);
void FreeClampParam(void **pdat);
void *GenerateEluParam(float32_t alpha);
void FreeEluParam(void **pdat);
void *GeneratePadParam(OnnxScalar value, int top, int bottom, int left, int right);
void FreePadParam(void **pdat);
void *GeneratePowParam(OnnxScalar exponent);
void FreePowParam(void **pdat);
void *GenerateFlipParam(int flip_axis0, int flip_axis1);
void FreeFlipParam(void **pdat);
/**
 * @brief only support 2-D tensor. start[i] == end[i] == 0 is not allowed.
 *
 * @param[in] naxes - slice axes number. The length of other inputs should be equal to naxes.
 * @param[in] axes - only support 0 or 1
 * @param[in] start - element start index (included)
 * @param[in] end - element end index (excluded)
 * @param[in] step - element step
 * @return void*
 */
void *GenerateSliceParam(int naxes, int *axes, int *start, int *end, int *step);
void FreeSliceParam(void **pdat);

/**
 * @brief 
 * 
 * @param[in] in_offset - The negative of the zero value for the input tensor
 * @param[in] out_offset - The negative of the zero value for the output tensor
 * @param[in] stride_w - kernel stride w
 * @param[in] stride_h - kernel stride h
 * @param[in] dilation_w - kernel dilation w
 * @param[in] dilation_h - kernel dilation h
 * @param[in] pad_w - kernel padding w
 * @param[in] pad_h - kernel padding h
 * @param[in] activation_min - min value
 * @param[in] activation_max - max value
 * @param[in] input - input tensor
 * @param[in] filter - input filter
 * @param[in] output - output tensor
 * @param[in] rvv - whether use rvv, the buffer size is different with or without rvv
 * @return void* ConvInteger private parameters
 */
void *GenerateConvIntegerParam(int32_t in_offset, int32_t out_offset, int32_t stride_w, int32_t stride_h, int32_t dilation_w, int32_t dilation_h,
                               int32_t pad_w, int32_t pad_h, int32_t activation_min, int32_t activation_max, const struct onnx_tensor_t *input,
                               const struct onnx_tensor_t *filter, const struct onnx_tensor_t *output, _Bool rvv);
void FreeConvIntegerParam(void **pdat);

/* ---------------- end of helper function ----------------- */

/* ---------------- start of operators ----------------- */

void BatchNormalization_float16(struct onnx_node_t *node);
void BatchNormalization_float16_rvv(struct onnx_node_t *node);
void BatchNormalization_float32(struct onnx_node_t *node);
void BatchNormalization_float32_rvv(struct onnx_node_t *node);

void LayerNormalization_float16(struct onnx_node_t *node);
void LayerNormalization_float16_rvv(struct onnx_node_t *node);
void LayerNormalization_float32(struct onnx_node_t *node);
void LayerNormalization_float32_rvv(struct onnx_node_t *node);

void RMSNormalization_float16(struct onnx_node_t *node);
void RMSNormalization_float16_rvv(struct onnx_node_t *node);
void RMSNormalization_float32(struct onnx_node_t *node);
void RMSNormalization_float32_rvv(struct onnx_node_t *node);

void Softmax_float16(struct onnx_node_t *node);
void Softmax_float16_rvv(struct onnx_node_t *node);
void Softmax_float32(struct onnx_node_t *node);
void Softmax_float32_rvv(struct onnx_node_t *node);

void Topk_int32(struct onnx_node_t *n);
void Topk_int32_rvv(struct onnx_node_t *n);

void MatMul_int8(struct onnx_node_t *node);
void MatMul_int8_rvv(struct onnx_node_t *node);
void MatMul_float16(struct onnx_node_t *node);
void MatMul_float16_rvv(struct onnx_node_t *node);
void MatMul_float32(struct onnx_node_t *node);
void MatMul_float32_rvv(struct onnx_node_t *node);

void Add_int8(struct onnx_node_t *node);
void Add_int8_rvv(struct onnx_node_t *node);
void Add_float16(struct onnx_node_t *node);
void Add_float16_rvv(struct onnx_node_t *node);
void Add_float32(struct onnx_node_t *node);
void Add_float32_rvv(struct onnx_node_t *node);

void Sub_int8(struct onnx_node_t *node);
void Sub_int8_rvv(struct onnx_node_t *node);
void Sub_float16(struct onnx_node_t *node);
void Sub_float16_rvv(struct onnx_node_t *node);
void Sub_float32(struct onnx_node_t *node);
void Sub_float32_rvv(struct onnx_node_t *node);

void Mul_int8(struct onnx_node_t *node);
void Mul_int8_rvv(struct onnx_node_t *node);
void Mul_float16(struct onnx_node_t *node);
void Mul_float16_rvv(struct onnx_node_t *node);
void Mul_float32(struct onnx_node_t *node);
void Mul_float32_rvv(struct onnx_node_t *node);

void Div_float16(struct onnx_node_t *node);
void Div_float16_rvv(struct onnx_node_t *node);
void Div_float32(struct onnx_node_t *node);
void Div_float32_rvv(struct onnx_node_t *node);

void Pow_float16(struct onnx_node_t *node);
void Pow_float16_rvv(struct onnx_node_t *node);
void Pow_float32(struct onnx_node_t *node);
void Pow_float32_rvv(struct onnx_node_t *node);

void Abs_int8(struct onnx_node_t *node);
void Abs_int8_rvv(struct onnx_node_t *node);
void Abs_int32(struct onnx_node_t *node);
void Abs_int32_rvv(struct onnx_node_t *node);
void Abs_float16(struct onnx_node_t *node);
void Abs_float16_rvv(struct onnx_node_t *node);
void Abs_float32(struct onnx_node_t *node);
void Abs_float32_rvv(struct onnx_node_t *node);

void Negate_int8(struct onnx_node_t *node);
void Negate_int8_rvv(struct onnx_node_t *node);
void Negate_int32(struct onnx_node_t *node);
void Negate_int32_rvv(struct onnx_node_t *node);
void Negate_float16(struct onnx_node_t *node);
void Negate_float16_rvv(struct onnx_node_t *node);
void Negate_float32(struct onnx_node_t *node);
void Negate_float32_rvv(struct onnx_node_t *node);

void Exp_float16(struct onnx_node_t *node);
void Exp_float16_rvv(struct onnx_node_t *node);
void Exp_float32(struct onnx_node_t *node);
void Exp_float32_rvv(struct onnx_node_t *node);

void Log_float16(struct onnx_node_t *node);
void Log_float16_rvv(struct onnx_node_t *node);
void Log_float32(struct onnx_node_t *node);
void Log_float32_rvv(struct onnx_node_t *node);

void Reciprocal_float16(struct onnx_node_t *node);
void Reciprocal_float16_rvv(struct onnx_node_t *node);
void Reciprocal_float32(struct onnx_node_t *node);
void Reciprocal_float32_rvv(struct onnx_node_t *node);

void Sqrt_float16(struct onnx_node_t *node);
void Sqrt_float16_rvv(struct onnx_node_t *node);
void Sqrt_float32(struct onnx_node_t *node);
void Sqrt_float32_rvv(struct onnx_node_t *node);

void Rsqrt_float16(struct onnx_node_t *node);
void Rsqrt_float16_rvv(struct onnx_node_t *node);
void Rsqrt_float32(struct onnx_node_t *node);
void Rsqrt_float32_rvv(struct onnx_node_t *node);

void Sin_float16(struct onnx_node_t *node);
void Sin_float16_rvv(struct onnx_node_t *node);
void Sin_float32(struct onnx_node_t *node);
void Sin_float32_rvv(struct onnx_node_t *node);

void Cos_float16(struct onnx_node_t *node);
void Cos_float16_rvv(struct onnx_node_t *node);
void Cos_float32(struct onnx_node_t *node);
void Cos_float32_rvv(struct onnx_node_t *node);

void Concat_int8(struct onnx_node_t *node);
void Concat_int8_rvv(struct onnx_node_t *node);
void Concat_int32(struct onnx_node_t *node);
void Concat_int32_rvv(struct onnx_node_t *node);
void Concat_float16(struct onnx_node_t *node);
void Concat_float16_rvv(struct onnx_node_t *node);
void Concat_float32(struct onnx_node_t *node);
void Concat_float32_rvv(struct onnx_node_t *node);

void Clamp_int8(struct onnx_node_t *node);
void Clamp_int8_rvv(struct onnx_node_t *node);
void Clamp_int32(struct onnx_node_t *node);
void Clamp_int32_rvv(struct onnx_node_t *node);
void Clamp_float16(struct onnx_node_t *node);
void Clamp_float16_rvv(struct onnx_node_t *node);
void Clamp_float32(struct onnx_node_t *node);
void Clamp_float32_rvv(struct onnx_node_t *node);

void Elu_float16(struct onnx_node_t *node);
void Elu_float16_rvv(struct onnx_node_t *node);
void Elu_float32(struct onnx_node_t *node);
void Elu_float32_rvv(struct onnx_node_t *node);

void Relu_float16(struct onnx_node_t *node);
void Relu_float16_rvv(struct onnx_node_t *node);
void Relu_float32(struct onnx_node_t *node);
void Relu_float32_rvv(struct onnx_node_t *node);

void Silu_float16(struct onnx_node_t *node);
void Silu_float16_rvv(struct onnx_node_t *node);
void Silu_float32(struct onnx_node_t *node);
void Silu_float32_rvv(struct onnx_node_t *node);

void Pad_int8(struct onnx_node_t *node);
void Pad_int8_rvv(struct onnx_node_t *node);
void Pad_int32(struct onnx_node_t *node);
void Pad_int32_rvv(struct onnx_node_t *node);
void Pad_float16(struct onnx_node_t *node);
void Pad_float16_rvv(struct onnx_node_t *node);
void Pad_float32(struct onnx_node_t *node);
void Pad_float32_rvv(struct onnx_node_t *node);

void Flip_int8(struct onnx_node_t *node);
void Flip_int8_rvv(struct onnx_node_t *node);
void Flip_int32(struct onnx_node_t *node);
void Flip_int32_rvv(struct onnx_node_t *node);
void Flip_float16(struct onnx_node_t *node);
void Flip_float16_rvv(struct onnx_node_t *node);
void Flip_float32(struct onnx_node_t *node);
void Flip_float32_rvv(struct onnx_node_t *node);

void Slice_int8(struct onnx_node_t *node);
void Slice_int8_rvv(struct onnx_node_t *node);
void Slice_int32(struct onnx_node_t *node);
void Slice_int32_rvv(struct onnx_node_t *node);
void Slice_float16(struct onnx_node_t *node);
void Slice_float16_rvv(struct onnx_node_t *node);
void Slice_float32(struct onnx_node_t *node);
void Slice_float32_rvv(struct onnx_node_t *node);

void Tile_int8(struct onnx_node_t *node);
void Tile_int8_rvv(struct onnx_node_t *node);
void Tile_int32(struct onnx_node_t *node);
void Tile_int32_rvv(struct onnx_node_t *node);
void Tile_float16(struct onnx_node_t *node);
void Tile_float16_rvv(struct onnx_node_t *node);
void Tile_float32(struct onnx_node_t *node);
void Tile_float32_rvv(struct onnx_node_t *node);

void GatherElements_int8(struct onnx_node_t *node);
void GatherElements_int8_rvv(struct onnx_node_t *node);
void GatherElements_int32(struct onnx_node_t *node);
void GatherElements_int32_rvv(struct onnx_node_t *node);
void GatherElements_float16(struct onnx_node_t *node);
void GatherElements_float16_rvv(struct onnx_node_t *node);
void GatherElements_float32(struct onnx_node_t *node);
void GatherElements_float32_rvv(struct onnx_node_t *node);

void ScatterElements_int8(struct onnx_node_t *node);
void ScatterElements_int8_rvv(struct onnx_node_t *node);
void ScatterElements_int32(struct onnx_node_t *node);
void ScatterElements_int32_rvv(struct onnx_node_t *node);
void ScatterElements_float16(struct onnx_node_t *node);
void ScatterElements_float16_rvv(struct onnx_node_t *node);
void ScatterElements_float32(struct onnx_node_t *node);
void ScatterElements_float32_rvv(struct onnx_node_t *node);

void ReduceAll(struct onnx_node_t *node);
void ReduceAll_rvv(struct onnx_node_t *node);
void ReduceAny(struct onnx_node_t *node);
void ReduceAny_rvv(struct onnx_node_t *node);

void ReduceMax_int8(struct onnx_node_t *node);
void ReduceMax_int8_rvv(struct onnx_node_t *node);
void ReduceMax_float16(struct onnx_node_t *node);
void ReduceMax_float16_rvv(struct onnx_node_t *node);
void ReduceMax_int32(struct onnx_node_t *node);
void ReduceMax_int32_rvv(struct onnx_node_t *node);
void ReduceMax_float32(struct onnx_node_t *n);
void ReduceMax_float32_rvv(struct onnx_node_t *n);

void ReduceMin_int8(struct onnx_node_t *node);
void ReduceMin_int8_rvv(struct onnx_node_t *node);
void ReduceMin_float16(struct onnx_node_t *node);
void ReduceMin_float16_rvv(struct onnx_node_t *node);
void ReduceMin_int32(struct onnx_node_t *node);
void ReduceMin_int32_rvv(struct onnx_node_t *node);
void ReduceMin_float32(struct onnx_node_t *n);
void ReduceMin_float32_rvv(struct onnx_node_t *n);

/* NOTE: Due to the accuracy of float type, multiplying float numbers
   not in order may lead to large deviations in the results */
void ReduceProd_float16(struct onnx_node_t *node);
void ReduceProd_float16_rvv(struct onnx_node_t *node);
void ReduceProd_float32(struct onnx_node_t *n);
void ReduceProd_float32_rvv(struct onnx_node_t *n);

void ReduceSum_float16(struct onnx_node_t *node);
void ReduceSum_float16_rvv(struct onnx_node_t *node);
void ReduceSum_float32(struct onnx_node_t *n);
void ReduceSum_float32_rvv(struct onnx_node_t *n);

int ConvInteger(struct onnx_node_t *n);
int ConvInteger_rvv(struct onnx_node_t *n);
/* ---------------- end of operators ----------------- */

#endif
