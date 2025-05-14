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
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
    float16_t v_float16;
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
    bfloat16_t v_bfloat16;
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
    float v_float32;
} OnnxScalar;

/* ---------------- start of helper function ----------------- */

/* 
 * Nuclei defined a CSR register for BF16 mode (-march=rv64imafdcv_xxlvfbf),
 * When the BFP16 mode bit is set, the f16 instruction is interpreted as bf16,
 * The BFP16 mode bit should be set when you test bf16 tests
 */
void csr_set_bf16_mode(void);
void csr_clr_bf16_mode(void);

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
void *GenerateGaussParam(float32_t alpha);
void FreeGaussParam(void **pdat);
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
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void BatchNormalization_float16(struct onnx_node_t *node);
void BatchNormalization_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void BatchNormalization_bfloat16(struct onnx_node_t *node);
void BatchNormalization_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void BatchNormalization_float32(struct onnx_node_t *node);
void BatchNormalization_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void LayerNormalization_float16(struct onnx_node_t *node);
void LayerNormalization_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void LayerNormalization_bfloat16(struct onnx_node_t *node);
void LayerNormalization_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void LayerNormalization_float32(struct onnx_node_t *node);
void LayerNormalization_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void BilinearInterpolation_float16(struct onnx_node_t *n);
void BilinearInterpolation_float16_rvv(struct onnx_node_t *n);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */

void BilinearInterpolation_float32(struct onnx_node_t *n);
void BilinearInterpolation_float32_rvv(struct onnx_node_t *n);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void RMSNormalization_float16(struct onnx_node_t *node);
void RMSNormalization_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void RMSNormalization_bfloat16(struct onnx_node_t *node);
void RMSNormalization_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void RMSNormalization_float32(struct onnx_node_t *node);
void RMSNormalization_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Softmax_float16(struct onnx_node_t *node);
void Softmax_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Softmax_bfloat16(struct onnx_node_t *node);
void Softmax_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Softmax_float32(struct onnx_node_t *node);
void Softmax_float32_rvv(struct onnx_node_t *node);

void Topk_int32(struct onnx_node_t *n);
void Topk_int32_rvv(struct onnx_node_t *n);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Topk_float16(struct onnx_node_t *n);
void Topk_float16_rvv(struct onnx_node_t *n);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Topk_bfloat16(struct onnx_node_t *n);
void Topk_bfloat16_rvv(struct onnx_node_t *n);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Topk_float32(struct onnx_node_t *n);
void Topk_float32_rvv(struct onnx_node_t *n);

void MatMul_int8(struct onnx_node_t *node);
void MatMul_int8_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void MatMul_float16(struct onnx_node_t *node);
void MatMul_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void MatMul_bfloat16(struct onnx_node_t *node);
void MatMul_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void MatMul_float32(struct onnx_node_t *node);
void MatMul_float32_rvv(struct onnx_node_t *node);

void Add_int8(struct onnx_node_t *node);
void Add_int8_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Add_float16(struct onnx_node_t *node);
void Add_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Add_bfloat16(struct onnx_node_t *node);
void Add_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Add_float32(struct onnx_node_t *node);
void Add_float32_rvv(struct onnx_node_t *node);

void Sub_int8(struct onnx_node_t *node);
void Sub_int8_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Sub_float16(struct onnx_node_t *node);
void Sub_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Sub_bfloat16(struct onnx_node_t *node);
void Sub_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Sub_float32(struct onnx_node_t *node);
void Sub_float32_rvv(struct onnx_node_t *node);

void Mul_int8(struct onnx_node_t *node);
void Mul_int8_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Mul_float16(struct onnx_node_t *node);
void Mul_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Mul_bfloat16(struct onnx_node_t *node);
void Mul_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Mul_float32(struct onnx_node_t *node);
void Mul_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Div_float16(struct onnx_node_t *node);
void Div_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Div_bfloat16(struct onnx_node_t *node);
void Div_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Div_float32(struct onnx_node_t *node);
void Div_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Pow_float16(struct onnx_node_t *node);
void Pow_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Pow_bfloat16(struct onnx_node_t *node);
void Pow_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Pow_float32(struct onnx_node_t *node);
void Pow_float32_rvv(struct onnx_node_t *node);

void Abs_int8(struct onnx_node_t *node);
void Abs_int8_rvv(struct onnx_node_t *node);
void Abs_int32(struct onnx_node_t *node);
void Abs_int32_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Abs_float16(struct onnx_node_t *node);
void Abs_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Abs_bfloat16(struct onnx_node_t *node);
void Abs_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Abs_float32(struct onnx_node_t *node);
void Abs_float32_rvv(struct onnx_node_t *node);

void Negate_int8(struct onnx_node_t *node);
void Negate_int8_rvv(struct onnx_node_t *node);
void Negate_int32(struct onnx_node_t *node);
void Negate_int32_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Negate_float16(struct onnx_node_t *node);
void Negate_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Negate_bfloat16(struct onnx_node_t *node);
void Negate_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Negate_float32(struct onnx_node_t *node);
void Negate_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Erf_float16(struct onnx_node_t *node);
void Erf_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
void Erf_float32(struct onnx_node_t *node);
void Erf_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Exp_float16(struct onnx_node_t *node);
void Exp_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Exp_bfloat16(struct onnx_node_t *node);
void Exp_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Exp_float32(struct onnx_node_t *node);
void Exp_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Log_float16(struct onnx_node_t *node);
void Log_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Log_bfloat16(struct onnx_node_t *node);
void Log_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Log_float32(struct onnx_node_t *node);
void Log_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Reciprocal_float16(struct onnx_node_t *node);
void Reciprocal_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Reciprocal_bfloat16(struct onnx_node_t *node);
void Reciprocal_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Reciprocal_float32(struct onnx_node_t *node);
void Reciprocal_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Sqrt_float16(struct onnx_node_t *node);
void Sqrt_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Sqrt_bfloat16(struct onnx_node_t *node);
void Sqrt_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Sqrt_float32(struct onnx_node_t *node);
void Sqrt_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Rsqrt_float16(struct onnx_node_t *node);
void Rsqrt_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Rsqrt_bfloat16(struct onnx_node_t *node);
void Rsqrt_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Rsqrt_float32(struct onnx_node_t *node);
void Rsqrt_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Sin_float16(struct onnx_node_t *node);
void Sin_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Sin_bfloat16(struct onnx_node_t *node);
void Sin_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Sin_float32(struct onnx_node_t *node);
void Sin_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Cos_float16(struct onnx_node_t *node);
void Cos_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Cos_bfloat16(struct onnx_node_t *node);
void Cos_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Cos_float32(struct onnx_node_t *node);
void Cos_float32_rvv(struct onnx_node_t *node);

void Concat_int8(struct onnx_node_t *node);
void Concat_int8_rvv(struct onnx_node_t *node);
void Concat_int32(struct onnx_node_t *node);
void Concat_int32_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Concat_float16(struct onnx_node_t *node);
void Concat_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Concat_bfloat16(struct onnx_node_t *node);
void Concat_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Concat_float32(struct onnx_node_t *node);
void Concat_float32_rvv(struct onnx_node_t *node);

void Clamp_int8(struct onnx_node_t *node);
void Clamp_int8_rvv(struct onnx_node_t *node);
void Clamp_int32(struct onnx_node_t *node);
void Clamp_int32_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Clamp_float16(struct onnx_node_t *node);
void Clamp_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Clamp_bfloat16(struct onnx_node_t *node);
void Clamp_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Clamp_float32(struct onnx_node_t *node);
void Clamp_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Elu_float16(struct onnx_node_t *node);
void Elu_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Elu_bfloat16(struct onnx_node_t *node);
void Elu_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Elu_float32(struct onnx_node_t *node);
void Elu_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Gauss_filter_float16(struct onnx_node_t *node);
void Gauss_filter_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
void Gauss_filter_float32(struct onnx_node_t *node);
void Gauss_filter_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Relu_float16(struct onnx_node_t *node);
void Relu_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Relu_bfloat16(struct onnx_node_t *node);
void Relu_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Relu_float32(struct onnx_node_t *node);
void Relu_float32_rvv(struct onnx_node_t *node);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Silu_float16(struct onnx_node_t *node);
void Silu_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Silu_bfloat16(struct onnx_node_t *node);
void Silu_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Silu_float32(struct onnx_node_t *node);
void Silu_float32_rvv(struct onnx_node_t *node);

void Pad_int8(struct onnx_node_t *node);
void Pad_int8_rvv(struct onnx_node_t *node);
void Pad_int32(struct onnx_node_t *node);
void Pad_int32_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Pad_float16(struct onnx_node_t *node);
void Pad_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Pad_bfloat16(struct onnx_node_t *node);
void Pad_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Pad_float32(struct onnx_node_t *node);
void Pad_float32_rvv(struct onnx_node_t *node);

void Flip_int8(struct onnx_node_t *node);
void Flip_int8_rvv(struct onnx_node_t *node);
void Flip_int32(struct onnx_node_t *node);
void Flip_int32_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Flip_float16(struct onnx_node_t *node);
void Flip_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Flip_bfloat16(struct onnx_node_t *node);
void Flip_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Flip_float32(struct onnx_node_t *node);
void Flip_float32_rvv(struct onnx_node_t *node);

void Slice_int8(struct onnx_node_t *node);
void Slice_int8_rvv(struct onnx_node_t *node);
void Slice_int32(struct onnx_node_t *node);
void Slice_int32_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Slice_float16(struct onnx_node_t *node);
void Slice_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Slice_bfloat16(struct onnx_node_t *node);
void Slice_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Slice_float32(struct onnx_node_t *node);
void Slice_float32_rvv(struct onnx_node_t *node);

void Tile_int8(struct onnx_node_t *node);
void Tile_int8_rvv(struct onnx_node_t *node);
void Tile_int32(struct onnx_node_t *node);
void Tile_int32_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void Tile_float16(struct onnx_node_t *node);
void Tile_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void Tile_bfloat16(struct onnx_node_t *node);
void Tile_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void Tile_float32(struct onnx_node_t *node);
void Tile_float32_rvv(struct onnx_node_t *node);

void GatherElements_int8(struct onnx_node_t *node);
void GatherElements_int8_rvv(struct onnx_node_t *node);
void GatherElements_int32(struct onnx_node_t *node);
void GatherElements_int32_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void GatherElements_float16(struct onnx_node_t *node);
void GatherElements_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void GatherElements_bfloat16(struct onnx_node_t *node);
void GatherElements_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void GatherElements_float32(struct onnx_node_t *node);
void GatherElements_float32_rvv(struct onnx_node_t *node);

void ScatterElements_int8(struct onnx_node_t *node);
void ScatterElements_int8_rvv(struct onnx_node_t *node);
void ScatterElements_int32(struct onnx_node_t *node);
void ScatterElements_int32_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void ScatterElements_float16(struct onnx_node_t *node);
void ScatterElements_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void ScatterElements_bfloat16(struct onnx_node_t *node);
void ScatterElements_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void ScatterElements_float32(struct onnx_node_t *node);
void ScatterElements_float32_rvv(struct onnx_node_t *node);

void ReduceAll(struct onnx_node_t *node);
void ReduceAll_rvv(struct onnx_node_t *node);
void ReduceAny(struct onnx_node_t *node);
void ReduceAny_rvv(struct onnx_node_t *node);

void ReduceMax_int8(struct onnx_node_t *node);
void ReduceMax_int8_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void ReduceMax_float16(struct onnx_node_t *node);
void ReduceMax_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void ReduceMax_bfloat16(struct onnx_node_t *node);
void ReduceMax_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void ReduceMax_int32(struct onnx_node_t *node);
void ReduceMax_int32_rvv(struct onnx_node_t *node);
void ReduceMax_float32(struct onnx_node_t *n);
void ReduceMax_float32_rvv(struct onnx_node_t *n);

void ReduceMin_int8(struct onnx_node_t *node);
void ReduceMin_int8_rvv(struct onnx_node_t *node);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void ReduceMin_float16(struct onnx_node_t *node);
void ReduceMin_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void ReduceMin_bfloat16(struct onnx_node_t *node);
void ReduceMin_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */

void ReduceMin_int32(struct onnx_node_t *node);
void ReduceMin_int32_rvv(struct onnx_node_t *node);
void ReduceMin_float32(struct onnx_node_t *n);
void ReduceMin_float32_rvv(struct onnx_node_t *n);

/* NOTE: Due to the accuracy of float type, multiplying float numbers
   not in order may lead to large deviations in the results */
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void ReduceProd_float16(struct onnx_node_t *node);
void ReduceProd_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void ReduceProd_bfloat16(struct onnx_node_t *node);
void ReduceProd_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void ReduceProd_float32(struct onnx_node_t *n);
void ReduceProd_float32_rvv(struct onnx_node_t *n);

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void ReduceSum_float16(struct onnx_node_t *node);
void ReduceSum_float16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void ReduceSum_bfloat16(struct onnx_node_t *node);
void ReduceSum_bfloat16_rvv(struct onnx_node_t *node);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void ReduceSum_float32(struct onnx_node_t *n);
void ReduceSum_float32_rvv(struct onnx_node_t *n);

int ConvInteger(struct onnx_node_t *n);
int ConvInteger_rvv(struct onnx_node_t *n);
/* ---------------- end of operators ----------------- */

#endif
