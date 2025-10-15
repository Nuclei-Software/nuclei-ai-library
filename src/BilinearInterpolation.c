/*
 * https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
 * https://github.com/xboot/libonnx/blob/master/src/default/BatchNormalization.c
 */

#include "operators.h"
#include "utils.h"

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void BilinearInterpolation_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *output = n->outputs[0];
    struct onnx_tensor_t *src = n->inputs[0];
    int8_t *poutput = (int8_t *)output->datas;
    int8_t *psrc = (int8_t *)src->datas;

    uint32_t src_width = src->dims[0];
    uint32_t src_height = src->dims[1];

    uint32_t target_width = output->dims[0];
    uint32_t target_height = output->dims[1];

    float16_t scale_x = (float16_t)src_width / target_width;
    float16_t scale_y = (float16_t)src_height / target_height;

    for (uint32_t y = 0; y < target_height; y++) {
        for (uint32_t x = 0; x < target_width; x++) {

            float16_t src_x = x * scale_x;
            float16_t src_y = y * scale_y;

            uint32_t x0 = (uint32_t)(src_x);
            uint32_t y0 = (uint32_t)(src_y);
            uint32_t x1 = (x0 + 1 < src_width) ? x0 + 1 : x0;
            uint32_t y1 = (y0 + 1 < src_height) ? y0 + 1 : y0;

            float16_t u = src_x - x0;
            float16_t v = src_y - y0;

            uint8_t q11 = psrc[y0 * src_width + x0];
            uint8_t q12 = psrc[y1 * src_width + x0];
            uint8_t q21 = psrc[y0 * src_width + x1];
            uint8_t q22 = psrc[y1 * src_width + x1];

            float16_t value = (1 - u) * (1 - v) * q11 +
                               u * (1 - v) * q21 +
                              (1 - u) * v * q12 +
                              u * v * q22;

            poutput[y * target_width + x] = (uint8_t)value;
        }
    }
}

void BilinearInterpolation_float16_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *output = n->outputs[0];
    struct onnx_tensor_t *src = n->inputs[0];
    int8_t *poutput = (int8_t *)output->datas;
    int8_t *psrc = (int8_t *)src->datas;

    uint32_t src_width = src->dims[0];
    uint32_t src_height = src->dims[1];

    uint32_t target_width = output->dims[0];
    uint32_t target_height = output->dims[1];

    float16_t scale_x = (float16_t)src_width / target_width;
    float16_t scale_y = (float16_t)src_height / target_height;

    for (uint32_t y = 0; y < target_height; y++) {
        float16_t src_y = y * scale_y;
        uint32_t y0 = (uint32_t)(src_y);
        uint32_t y1 = (y0 + 1 < src_height) ? y0 + 1 : y0;
        float16_t v = src_y - y0;

        size_t vl;
        size_t blkCnt = target_width;

        vl = __riscv_vsetvlmax_e16m8();
        vuint16m8_t vx_b = __riscv_vid_v_u16m8(vl);
        uint32_t offset = 0;
        uint8_t *pDes = poutput + y * target_width;

        for (; (vl = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= vl) {
            vuint16m8_t vx = __riscv_vadd_vx_u16m8(vx_b, offset, vl);
            offset += vl;
            vfloat16m8_t vsrc_x = __riscv_vfmul_vf_f16m8 (__riscv_vfcvt_f_xu_v_f16m8(vx, vl), scale_x, vl);
            vuint16m8_t x0 = __riscv_vfcvt_rtz_xu_f_v_u16m8(vsrc_x, vl);
            vuint16m8_t x1 = __riscv_vminu_vx_u16m8(__riscv_vadd_vx_u16m8(x0, 1, vl), src_width - 1, vl);
            vfloat16m8_t vu = __riscv_vfsub_vv_f16m8(vsrc_x, __riscv_vfcvt_f_xu_v_f16m8(x0, vl), vl);
    
            vuint8m4_t vq11 = __riscv_vloxei16_v_u8m4 (psrc + y0 * src_width, x0, vl);
            vuint8m4_t vq12 = __riscv_vloxei16_v_u8m4 (psrc + y1 * src_width, x0, vl);
            vuint8m4_t vq21 = __riscv_vloxei16_v_u8m4 (psrc + y0 * src_width, x1, vl);
            vuint8m4_t vq22 = __riscv_vloxei16_v_u8m4 (psrc + y1 * src_width, x1, vl);

            vfloat16m8_t value1 = __riscv_vfwcvt_f_xu_v_f16m8(vq11, vl);
            value1 = __riscv_vfmul_vf_f16m8(value1, 1-v, vl);
            value1 = __riscv_vfmul_vv_f16m8(value1, __riscv_vfrsub_vf_f16m8(vu, 1.0, vl), vl);

            vfloat16m8_t value2 = __riscv_vfwcvt_f_xu_v_f16m8(vq21, vl);
            value2 = __riscv_vfmul_vf_f16m8(value2, 1-v, vl);
            value2 = __riscv_vfmul_vv_f16m8(value2, vu, vl);
            value1 = __riscv_vfadd_vv_f16m8(value1, value2, vl);

            vfloat16m8_t value3 = __riscv_vfwcvt_f_xu_v_f16m8(vq12, vl);
            value3 = __riscv_vfmul_vf_f16m8(value3, v, vl);
            value3 = __riscv_vfmul_vv_f16m8(value3, __riscv_vfrsub_vf_f16m8(vu, 1.0, vl), vl);
            value1 = __riscv_vfadd_vv_f16m8(value1, value3, vl);

            vfloat16m8_t value4 = __riscv_vfwcvt_f_xu_v_f16m8(vq22, vl);
            value4 = __riscv_vfmul_vf_f16m8(value4, v, vl);
            value4 = __riscv_vfmul_vv_f16m8(value4, vu, vl);
            value1 = __riscv_vfadd_vv_f16m8(value1, value4, vl);

            __riscv_vse8_v_u8m4(pDes, __riscv_vfncvt_rtz_xu_f_w_u8m4(value1, vl), vl);
            pDes += vl;
        }
    }
}
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */

void BilinearInterpolation_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *output = n->outputs[0];
    struct onnx_tensor_t *src = n->inputs[0];
    int8_t *poutput = (int8_t *)output->datas;
    int8_t *psrc = (int8_t *)src->datas;

    uint32_t src_width = src->dims[0];
    uint32_t src_height = src->dims[1];

    uint32_t target_width = output->dims[0];
    uint32_t target_height = output->dims[1];

    float32_t scale_x = (float32_t)src_width / target_width;
    float32_t scale_y = (float32_t)src_height / target_height;

    for (uint32_t y = 0; y < target_height; y++) {
        for (uint32_t x = 0; x < target_width; x++) {

            float32_t src_x = x * scale_x;
            float32_t src_y = y * scale_y;

            uint32_t x0 = (uint32_t)(src_x);
            uint32_t y0 = (uint32_t)(src_y);
            uint32_t x1 = (x0 + 1 < src_width) ? x0 + 1 : x0;
            uint32_t y1 = (y0 + 1 < src_height) ? y0 + 1 : y0;

            float32_t u = src_x - x0;
            float32_t v = src_y - y0;

            uint8_t q11 = psrc[y0 * src_width + x0];
            uint8_t q12 = psrc[y1 * src_width + x0];
            uint8_t q21 = psrc[y0 * src_width + x1];
            uint8_t q22 = psrc[y1 * src_width + x1];

            float32_t value = (1 - u) * (1 - v) * q11 +
                               u * (1 - v) * q21 +
                              (1 - u) * v * q12 +
                              u * v * q22;

            poutput[y * target_width + x] = (uint8_t)value;
        }
    }
}

void BilinearInterpolation_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *output = n->outputs[0];
    struct onnx_tensor_t *src = n->inputs[0];
    int8_t *poutput = (int8_t *)output->datas;
    int8_t *psrc = (int8_t *)src->datas;

    uint32_t src_width = src->dims[0];
    uint32_t src_height = src->dims[1];

    uint32_t target_width = output->dims[0];
    uint32_t target_height = output->dims[1];

    float32_t scale_x = (float32_t)src_width / target_width;
    float32_t scale_y = (float32_t)src_height / target_height;

    for (uint32_t y = 0; y < target_height; y++) {
        float32_t src_y = y * scale_y;
        uint32_t y0 = (uint32_t)(src_y);
        uint32_t y1 = (y0 + 1 < src_height) ? y0 + 1 : y0;
        float32_t v = src_y - y0;

        size_t vl;
        size_t blkCnt = target_width;

        vl = __riscv_vsetvlmax_e32m8();
        vuint32m8_t vx_b = __riscv_vid_v_u32m8(vl);
        uint32_t offset = 0;
        uint8_t *pDes = poutput + y * target_width;

        for (; (vl = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= vl) {
            vuint32m8_t vx = __riscv_vadd_vx_u32m8(vx_b, offset, vl);
            offset += vl;
            vfloat32m8_t vsrc_x = __riscv_vfmul_vf_f32m8 (__riscv_vfcvt_f_xu_v_f32m8(vx, vl), scale_x, vl);
            vuint32m8_t x0 = __riscv_vfcvt_rtz_xu_f_v_u32m8(vsrc_x, vl);
            vuint32m8_t x1 = __riscv_vminu_vx_u32m8(__riscv_vadd_vx_u32m8(x0, 1, vl), src_width - 1, vl);
            vfloat32m8_t vu = __riscv_vfsub_vv_f32m8(vsrc_x, __riscv_vfcvt_f_xu_v_f32m8(x0, vl), vl);

            vuint8m2_t vq11 = __riscv_vloxei32_v_u8m2 (psrc + y0 * src_width, x0, vl);
            vuint8m2_t vq12 = __riscv_vloxei32_v_u8m2 (psrc + y1 * src_width, x0, vl);
            vuint8m2_t vq21 = __riscv_vloxei32_v_u8m2 (psrc + y0 * src_width, x1, vl);
            vuint8m2_t vq22 = __riscv_vloxei32_v_u8m2 (psrc + y1 * src_width, x1, vl);

            vfloat32m8_t value1 = __riscv_vfwcvt_f_xu_v_f32m8(__riscv_vwaddu_vx_u16m4(vq11, 0, vl), vl);
            value1 = __riscv_vfmul_vf_f32m8(value1, 1-v, vl);
            value1 = __riscv_vfmul_vv_f32m8(value1, __riscv_vfrsub_vf_f32m8(vu, 1.0, vl), vl);

            vfloat32m8_t value2 = __riscv_vfwcvt_f_xu_v_f32m8(__riscv_vwaddu_vx_u16m4(vq21, 0, vl), vl);
            value2 = __riscv_vfmul_vf_f32m8(value2, 1-v, vl);
            value2 = __riscv_vfmul_vv_f32m8(value2, vu, vl);
            value1 = __riscv_vfadd_vv_f32m8(value1, value2, vl);

            vfloat32m8_t value3 = __riscv_vfwcvt_f_xu_v_f32m8(__riscv_vwaddu_vx_u16m4(vq12, 0, vl), vl);
            value3 = __riscv_vfmul_vf_f32m8(value3, v, vl);
            value3 = __riscv_vfmul_vv_f32m8(value3, __riscv_vfrsub_vf_f32m8(vu, 1.0, vl), vl);
            value1 = __riscv_vfadd_vv_f32m8(value1, value3, vl);

            vfloat32m8_t value4 = __riscv_vfwcvt_f_xu_v_f32m8(__riscv_vwaddu_vx_u16m4(vq22, 0, vl), vl);
            value4 = __riscv_vfmul_vf_f32m8(value4, v, vl);
            value4 = __riscv_vfmul_vv_f32m8(value4, vu, vl);
            value1 = __riscv_vfadd_vv_f32m8(value1, value4, vl);

            vuint8m2_t vres = __riscv_vncvt_x_x_w_u8m2(__riscv_vfncvt_rtz_xu_f_w_u16m4(value1, vl), vl);
            __riscv_vse8_v_u8m2(pDes, vres, vl);
            pDes += vl;
        }
    }
}