/*
 * https://onnx.ai/onnx/operators/onnx__Cos.html#cos
 * https://github.com/xboot/libonnx/blob/master/src/default/Cos.c
 */

#include "operators.h"

void Cos_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];

    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        py[i] = (float16_t)cosf((float32_t)px[i]);
    }
}

void Cos_float16_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];

    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;

    size_t vblkCnt = y->ndata; /* Loop counter */
    size_t vl;
    vfloat32m8_t vx, vy, vz;
    vint32m8_t vx_int;
    vbool4_t mask;
    // 1 - 1/2! x^2 + 1/4! x^4 - 1/6! x^6 + 1/8! x^8 - 1/10! x^10...
    for (; (vl = __riscv_vsetvl_e32m8(vblkCnt)) > 0; vblkCnt -= vl) {
        // Note: Because of the accuracy of float16, should use float32
        vx = __riscv_vfwadd_vf_f32m8(__riscv_vle16_v_f16m4(px, vl), 0.0, vl);
        px += vl;
        vx = __riscv_vfmul_vf_f32m8(vx, 1.0 / (2 * PI), vl);
        vx_int = __riscv_vfcvt_rtz_x_f_v_i32m8(vx, vl);
        vx = __riscv_vfsub_vv_f32m8(vx, __riscv_vfcvt_f_x_v_f32m8(vx_int, vl), vl);
        vx = __riscv_vfmul_vf_f32m8(vx, 2 * PI, vl);

        mask = __riscv_vmfgt_vf_f32m8_b4(vx, PI, vl);
        vx = __riscv_vfadd_vf_f32m8_mu(mask, vx, vx, -2 * PI, vl);

        mask = __riscv_vmflt_vf_f32m8_b4(vx, -PI, vl);
        vx = __riscv_vfadd_vf_f32m8_mu(mask, vx, vx, 2 * PI, vl);

        vy = __riscv_vfmul_vv_f32m8(vx, vx, vl);
        vz = __riscv_vfmul_vf_f32m8(vy, -1.0 / 3628800, vl);                              // 1/10!
        vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, 1.0 / 40320, vl), vl); // 1/8!
        vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, -1.0 / 720, vl), vl);  // 1/6!
        vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, 1.0 / 24, vl), vl);    // 1/4!
        vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, -1.0 / 2, vl), vl);    // 1/2!
        vz = __riscv_vfadd_vf_f32m8(vz, 1, vl);

        __riscv_vse16_v_f16m4(py, __riscv_vfncvt_f_f_w_f16m4(vz, vl), vl);
        py += vl;
    }
}

void Cos_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];

    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        py[i] = cosf(px[i]);
    }
}

void Cos_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];

    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;

    size_t vblkCnt = y->ndata; /* Loop counter */
    size_t vl;
    vfloat32m8_t vx, vy, vz;
    vint32m8_t vx_int;
    vbool4_t mask;
    // 1 - 1/2! x^2 + 1/4! x^4 - 1/6! x^6 + 1/8! x^8 - 1/10! x^10...
    for (; (vl = __riscv_vsetvl_e32m8(vblkCnt)) > 0; vblkCnt -= vl) {
        vx = __riscv_vle32_v_f32m8(px, vl);
        px += vl;
        vx = __riscv_vfmul_vf_f32m8(vx, 1.0 / (2 * PI), vl);
        vx_int = __riscv_vfcvt_rtz_x_f_v_i32m8(vx, vl);
        vx = __riscv_vfsub_vv_f32m8(vx, __riscv_vfcvt_f_x_v_f32m8(vx_int, vl), vl);
        vx = __riscv_vfmul_vf_f32m8(vx, 2 * PI, vl);

        mask = __riscv_vmfgt_vf_f32m8_b4(vx, PI, vl);
        vx = __riscv_vfadd_vf_f32m8_mu(mask, vx, vx, -2 * PI, vl);

        mask = __riscv_vmflt_vf_f32m8_b4(vx, -PI, vl);
        vx = __riscv_vfadd_vf_f32m8_mu(mask, vx, vx, 2 * PI, vl);

        vy = __riscv_vfmul_vv_f32m8(vx, vx, vl);
        vz = __riscv_vfmul_vf_f32m8(vy, -1.0 / 3628800, vl);                              // 1/10!
        vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, 1.0 / 40320, vl), vl); // 1/8!
        vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, -1.0 / 720, vl), vl);  // 1/6!
        vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, 1.0 / 24, vl), vl);    // 1/4!
        vz = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(vz, -1.0 / 2, vl), vl);    // 1/2!
        vz = __riscv_vfadd_vf_f32m8(vz, 1, vl);

        __riscv_vse32_v_f32m8(py, vz, vl);
        py += vl;
    }
}
