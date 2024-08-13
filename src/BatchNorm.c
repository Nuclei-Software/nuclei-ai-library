/**
 * @file BatchNormalization.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

/*
 * https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
 * https://github.com/xboot/libonnx/blob/master/src/default/BatchNormalization.c
 */

#include "operators.h"
#include "utils.h"

struct operator_pdata_t {
    float epsilon;
    float momentum;
};

void BatchNormalization_float16(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *scale = n->inputs[1];
    struct onnx_tensor_t *b = n->inputs[2];
    struct onnx_tensor_t *mean = n->inputs[3];
    struct onnx_tensor_t *var = n->inputs[4];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *pscale = (float16_t *)scale->datas;
    float16_t *pb = (float16_t *)b->datas;
    float16_t *pmean = (float16_t *)mean->datas;
    float16_t *pvar = (float16_t *)var->datas;
    float16_t *py = (float16_t *)y->datas;
    int N = x->dims[0];
    int C = x->dims[1];
    int NC = N * C;
    int channel = 1;
    int i, j, o, jc;

    for (i = 2; i < x->ndim; i++)
        channel *= x->dims[i];
    for (j = 0; j < NC; j++) {
        o = j * channel;
        jc = j % C;
        for (i = 0; i < channel; i++)
            py[o + i] = pscale[jc] * ((px[o + i] - pmean[jc]) / sqrtf(pvar[jc] + pdat->epsilon)) + pb[jc];
    }
}

void BatchNormalization_float16_rvv(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *scale = n->inputs[1];
    struct onnx_tensor_t *b = n->inputs[2];
    struct onnx_tensor_t *mean = n->inputs[3];
    struct onnx_tensor_t *var = n->inputs[4];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *pscale = (float16_t *)scale->datas;
    float16_t *pb = (float16_t *)b->datas;
    float16_t *pmean = (float16_t *)mean->datas;
    float16_t *pvar = (float16_t *)var->datas;
    float16_t *py = (float16_t *)y->datas;
    int N = x->dims[0];
    int C = x->dims[1];
    int NC = N * C;
    int channel = 1;
    int i, j, o, jc;

    for (i = 2; i < x->ndim; i++)
        channel *= x->dims[i];
    for (j = 0; j < NC; j++) {
        o = j * channel;
        jc = j % C;

        size_t blkCnt = channel;
        size_t vl;
        vfloat16m8_t vx;
        float16_t *pSrc = px + o;
        float16_t *pDst = py + o;
        float16_t vscale = pscale[jc];
        float16_t vmean = pmean[jc];
        float16_t vval = sqrtf(pvar[jc] + pdat->epsilon);
        float16_t vb = pb[jc];
        for (; (vl = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= vl) {
            vx = __riscv_vle16_v_f16m8(pSrc, vl);
            pSrc += vl;
            vx = __riscv_vfsub_vf_f16m8(vx, vmean, vl);
            vx = __riscv_vfmul_vf_f16m8(vx, vscale, vl);
            vx = __riscv_vfdiv_vf_f16m8(vx, vval, vl);
            vx = __riscv_vfadd_vf_f16m8(vx, vb, vl);
            __riscv_vse16_v_f16m8(pDst, vx, vl);
            pDst += vl;
        }
    }
}

void BatchNormalization_float32(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *scale = n->inputs[1];
    struct onnx_tensor_t *b = n->inputs[2];
    struct onnx_tensor_t *mean = n->inputs[3];
    struct onnx_tensor_t *var = n->inputs[4];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *pscale = (float32_t *)scale->datas;
    float32_t *pb = (float32_t *)b->datas;
    float32_t *pmean = (float32_t *)mean->datas;
    float32_t *pvar = (float32_t *)var->datas;
    float32_t *py = (float32_t *)y->datas;
    int N = x->dims[0];
    int C = x->dims[1];
    int NC = N * C;
    int channel = 1;
    int i, j, o, jc;

    for (i = 2; i < x->ndim; i++)
        channel *= x->dims[i];
    for (j = 0; j < NC; j++) {
        o = j * channel;
        jc = j % C;
        for (i = 0; i < channel; i++)
            py[o + i] = pscale[jc] * ((px[o + i] - pmean[jc]) / sqrtf(pvar[jc] + pdat->epsilon)) + pb[jc];
    }
}

void BatchNormalization_float32_rvv(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *scale = n->inputs[1];
    struct onnx_tensor_t *b = n->inputs[2];
    struct onnx_tensor_t *mean = n->inputs[3];
    struct onnx_tensor_t *var = n->inputs[4];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *pscale = (float32_t *)scale->datas;
    float32_t *pb = (float32_t *)b->datas;
    float32_t *pmean = (float32_t *)mean->datas;
    float32_t *pvar = (float32_t *)var->datas;
    float32_t *py = (float32_t *)y->datas;
    int N = x->dims[0];
    int C = x->dims[1];
    int NC = N * C;
    int channel = 1;
    int i, j, o, jc;

    for (i = 2; i < x->ndim; i++)
        channel *= x->dims[i];
    for (j = 0; j < NC; j++) {
        o = j * channel;
        jc = j % C;
        size_t blkCnt = channel;
        size_t vl;
        vfloat32m8_t vx;
        float32_t *pSrc = px + o;
        float32_t *pDst = py + o;
        float32_t vscale = pscale[jc];
        float32_t vmean = pmean[jc];
        float32_t vval = sqrtf(pvar[jc] + pdat->epsilon);
        float32_t vb = pb[jc];
        for (; (vl = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= vl) {
            vx = __riscv_vle32_v_f32m8(pSrc, vl);
            pSrc += vl;
            vx = __riscv_vfsub_vf_f32m8(vx, vmean, vl);
            vx = __riscv_vfmul_vf_f32m8(vx, vscale, vl);
            vx = __riscv_vfdiv_vf_f32m8(vx, vval, vl);
            vx = __riscv_vfadd_vf_f32m8(vx, vb, vl);
            __riscv_vse32_v_f32m8(pDst, vx, vl);
            pDst += vl;
        }
    }
}

void *GenerateBatchNormParam(float epsilon, float momentum)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)MALLOC_ASSERT(sizeof(struct operator_pdata_t));
    pdat->epsilon = epsilon;
    pdat->momentum = momentum;
    return pdat;
}

void FreeBatchNormParam(void **pdat)
{
    free(*pdat);
    *pdat = NULL;
}