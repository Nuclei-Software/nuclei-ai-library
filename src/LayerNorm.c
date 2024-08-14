/*
 * https://onnx.ai/onnx/operators/onnx__LayerNormalization.html
 *
 */

#include "operators.h"
#include "utils.h"

struct operator_pdata_t {
    float epsilon;
    float momentum;
};

void LayerNormalization_float16(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    int N = x->dims[0];
    int D = x->dims[1];

    for (int i = 0; i < N; i++) {
        float16_t mean = 0;
        for (int j = 0; j < D; j++) {
            mean += px[i * D + j];
        }
        mean /= D;

        float16_t variance = 0;
        for (int j = 0; j < D; j++) {
            variance += (px[i * D + j] - mean) * (px[i * D + j] - mean);
        }
        variance /= D;

        float16_t inv_stddev = 1.0 / sqrtf((float32_t)(variance + pdat->epsilon));

        for (int j = 0; j < D; j++) {
            py[i * D + j] = (px[i * D + j] - mean) * inv_stddev;
        }
    }
}

void LayerNormalization_float16_rvv(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    int N = x->dims[0];
    int D = x->dims[1];

    for (int i = 0; i < N; i++) {
        float16_t mean = 0;
        float16_t variance = 0;

        size_t blkCnt = D;
        size_t l;
        vfloat16m8_t vx;
        vfloat16m1_t vsum;
        float16_t *pSrc = px + i * D;
        vsum = __riscv_vfmv_v_f_f16m1(0.0, 1);
        for (; (l = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= l) {
            vx = __riscv_vle16_v_f16m8(pSrc, l);
            pSrc += l;
            vsum = __riscv_vfredusum_vs_f16m8_f16m1(vx, vsum, l);
        }
        mean = __riscv_vfmv_f_s_f16m1_f16(vsum);
        mean /= D;

        blkCnt = D;
        pSrc = px + i * D;
        vsum = __riscv_vfmv_v_f_f16m1(0.0, 1);
        for (; (l = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= l) {
            vx = __riscv_vle16_v_f16m8(pSrc, l);
            pSrc += l;
            vx = __riscv_vfsub_vf_f16m8(vx, mean, l);
            vsum = __riscv_vfredusum_vs_f16m8_f16m1(__riscv_vfmul_vv_f16m8(vx, vx, l), vsum, l);
        }
        variance = __riscv_vfmv_f_s_f16m1_f16(vsum);
        variance /= D;

        float16_t inv_stddev = 1.0 / sqrtf((float32_t)(variance + pdat->epsilon));
        blkCnt = D;
        pSrc = px + i * D;
        float16_t *pDes = py + i * D;
        for (; (l = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= l) {
            vx = __riscv_vle16_v_f16m8(pSrc, l);
            pSrc += l;
            vx = __riscv_vfsub_vf_f16m8(vx, mean, l);
            vx = __riscv_vfmul_vf_f16m8(vx, inv_stddev, l);
            __riscv_vse16_v_f16m8(pDes, vx, l);
            pDes += l;
        }
    }
}

void LayerNormalization_float32(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    int N = x->dims[0];
    int D = x->dims[1];

    for (int i = 0; i < N; i++) {
        float32_t mean = 0.0f;
        for (int j = 0; j < D; j++) {
            mean += px[i * D + j];
        }
        mean /= D;

        float32_t variance = 0.0f;
        for (int j = 0; j < D; j++) {
            variance += (px[i * D + j] - mean) * (px[i * D + j] - mean);
        }
        variance /= D;

        float32_t inv_stddev = 1.0 / sqrtf(variance + pdat->epsilon);

        for (int j = 0; j < D; j++) {
            py[i * D + j] = (px[i * D + j] - mean) * inv_stddev;
        }
    }
}

void LayerNormalization_float32_rvv(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    int N = x->dims[0];
    int D = x->dims[1];

    for (int i = 0; i < N; i++) {
        float32_t mean = 0.0f;
        float32_t variance = 0.0f;

        size_t blkCnt = D;
        size_t l;
        vfloat32m8_t vx;
        vfloat32m1_t vsum;
        float32_t *pSrc = px + i * D;
        vsum = __riscv_vfmv_v_f_f32m1(0.0, 1);
        for (; (l = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= l) {
            vx = __riscv_vle32_v_f32m8(pSrc, l);
            pSrc += l;
            vsum = __riscv_vfredusum_vs_f32m8_f32m1(vx, vsum, l);
        }
        mean = __riscv_vfmv_f_s_f32m1_f32(vsum);
        mean /= D;

        blkCnt = D;
        pSrc = px + i * D;
        vsum = __riscv_vfmv_v_f_f32m1(0.0, 1);
        for (; (l = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= l) {
            vx = __riscv_vle32_v_f32m8(pSrc, l);
            pSrc += l;
            vx = __riscv_vfsub_vf_f32m8(vx, mean, l);
            vsum = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(vx, vx, l), vsum, l);
        }
        variance = __riscv_vfmv_f_s_f32m1_f32(vsum);
        variance /= D;

        float32_t inv_stddev = 1.0 / sqrtf(variance + pdat->epsilon);
        blkCnt = D;
        pSrc = px + i * D;
        float32_t *pDes = py + i * D;
        for (; (l = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= l) {
            vx = __riscv_vle32_v_f32m8(pSrc, l);
            pSrc += l;
            vx = __riscv_vfsub_vf_f32m8(vx, mean, l);
            vx = __riscv_vfmul_vf_f32m8(vx, inv_stddev, l);
            __riscv_vse32_v_f32m8(pDes, vx, l);
            pDes += l;
        }
    }
}

void *GenerateLayerNormParam(float epsilon, float momentum)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)MALLOC_ASSERT(sizeof(struct operator_pdata_t));
    pdat->epsilon = epsilon;
    pdat->momentum = momentum;
    return pdat;
}

void FreeLayerNormParam(void **pdat)
{
    free(*pdat);
    *pdat = NULL;
}