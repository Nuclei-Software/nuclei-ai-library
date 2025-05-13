/*
 * https://pytorch.org/docs/stable/special.html#torch.special.erf
 * https://github.com/xboot/libonnx/blob/master/src/default/Erf.c
 */

#include "operators.h"

void Erf_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = (float16_t)erff((float32_t)px[i]);
}

#define a1 0.0705230784
#define a2 0.0422820123
#define a3 0.0092705272
#define a4 0.0001520143
#define a5 0.0002765672
#define a6 0.0000430638

/*************************************************************************************
 * erf(x) = 1 - 1 / (1 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 + a6*x^6)^16 (x >= 0)
 **************************************************************************************/

void Erf_float16_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t l;

    for (; (l = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= l) {

        vfloat16m8_t _x = __riscv_vle16_v_f16m8(px, l);
        px += l;
        vbool2_t _mask = __riscv_vmflt_vf_f16m8_b2(_x, 0.0f, l);
        _x = __riscv_vfmul_vf_f16m8_m(_mask, _x, -1.0f, l);

        vfloat16m8_t xx = __riscv_vfmul_vf_f16m8(_x, a6, l);
        xx = __riscv_vfadd_vf_f16m8(xx, a5, l);
        xx = __riscv_vfmul_vv_f16m8(xx, _x, l);
        xx = __riscv_vfadd_vf_f16m8(xx, a4, l);
        xx = __riscv_vfmul_vv_f16m8(xx, _x, l);
        xx = __riscv_vfadd_vf_f16m8(xx, a3, l);
        xx = __riscv_vfmul_vv_f16m8(xx, _x, l);
        xx = __riscv_vfadd_vf_f16m8(xx, a2, l);
        xx = __riscv_vfmul_vv_f16m8(xx, _x, l);
        xx = __riscv_vfadd_vf_f16m8(xx, a1, l);
        xx = __riscv_vfmul_vv_f16m8(xx, _x, l);
        xx = __riscv_vfadd_vf_f16m8(xx, 1.0, l);

        xx = __riscv_vfmul_vv_f16m8(xx, xx, l);
        xx = __riscv_vfmul_vv_f16m8(xx, xx, l);
        xx = __riscv_vfmul_vv_f16m8(xx, xx, l);
        xx = __riscv_vfmul_vv_f16m8(xx, xx, l);

        vfloat16m8_t _y = __riscv_vfrdiv_vf_f16m8(xx, -1.0f, l);
        _y = __riscv_vfadd_vf_f16m8(_y, 1.0f, l);
        _y = __riscv_vfmul_vf_f16m8_m(_mask, _y, -1.0f, l);

        __riscv_vse16_v_f16m8(py, _y, l);
        py += l;
    }
}


void Erf_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = erff(px[i]);
}

#define a1 0.0705230784
#define a2 0.0422820123
#define a3 0.0092705272
#define a4 0.0001520143
#define a5 0.0002765672
#define a6 0.0000430638

/*************************************************************************************
 * erf(x) = 1 - 1 / (1 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 + a6*x^6)^16 (x>=0)
 **************************************************************************************/

void Erf_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t l;

    for (; (l = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= l) {

        vfloat32m8_t _x = __riscv_vle32_v_f32m8(px, l);
        px += l;
        vbool4_t _mask = __riscv_vmflt_vf_f32m8_b4(_x, 0.0f, l);
        _x = __riscv_vfmul_vf_f32m8_m(_mask, _x, -1.0f, l);

        vfloat32m8_t xx = __riscv_vfmul_vf_f32m8(_x, a6, l);
        xx = __riscv_vfadd_vf_f32m8(xx, a5, l);
        xx = __riscv_vfmul_vv_f32m8(xx, _x, l);
        xx = __riscv_vfadd_vf_f32m8(xx, a4, l);
        xx = __riscv_vfmul_vv_f32m8(xx, _x, l);
        xx = __riscv_vfadd_vf_f32m8(xx, a3, l);
        xx = __riscv_vfmul_vv_f32m8(xx, _x, l);
        xx = __riscv_vfadd_vf_f32m8(xx, a2, l);
        xx = __riscv_vfmul_vv_f32m8(xx, _x, l);
        xx = __riscv_vfadd_vf_f32m8(xx, a1, l);
        xx = __riscv_vfmul_vv_f32m8(xx, _x, l);
        xx = __riscv_vfadd_vf_f32m8(xx, 1.0, l);

        xx = __riscv_vfmul_vv_f32m8(xx, xx, l);
        xx = __riscv_vfmul_vv_f32m8(xx, xx, l);
        xx = __riscv_vfmul_vv_f32m8(xx, xx, l);
        xx = __riscv_vfmul_vv_f32m8(xx, xx, l);

        vfloat32m8_t _y = __riscv_vfrdiv_vf_f32m8(xx, -1.0f, l);
        _y = __riscv_vfadd_vf_f32m8(_y, 1.0f, l);
        _y = __riscv_vfmul_vf_f32m8_m(_mask, _y, -1.0f, l);

        __riscv_vse32_v_f32m8(py, _y, l);
        py += l;
    }
}