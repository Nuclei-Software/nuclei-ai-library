/*
 * https://onnx.ai/onnx/operators/onnx__Pow.html
 * https://github.com/xboot/libonnx/blob/master/src/default/Pow.c
 */

// TODO(shuzhuo): exponent can be either a single float number, not support a Tensor

#include "operators.h"
#include "utils.h"

struct operator_pdata_t {
    union onnx_scalar_t exponent;
};

void Pow_float16(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    float16_t exponent = pdat->exponent.v_float16;
    for (size_t i = 0, l = y->ndata; i < l; i++) {
        py[i] = (float16_t)pow((float32_t)px[i], (float32_t)exponent);
    }
}

void Pow_float16_rvv(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    float16_t exponent = pdat->exponent.v_float16;

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t vl;
    vfloat16m8_t vx, vy, vz, v1;
    vint16m8_t vx_int, vk;
    for (; (vl = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= vl) {
        // lnx
        vx = __riscv_vle16_v_f16m8(px, vl);
        px += vl;
        vx_int = __riscv_vreinterpret_v_f16m8_i16m8(vx);
        vk = __riscv_vsra_vx_i16m8(__riscv_vsub_vx_i16m8(vx_int, (15 << 10), vl), 10, vl);

        vz = __riscv_vfadd_vf_f16m8(__riscv_vfcvt_f_x_v_f16m8(vk, vl), 0.5f, vl);
        vz = __riscv_vfmul_vf_f16m8(vz, 0.6931471805599453f, vl); // ln2

        vx_int = __riscv_vor_vx_i16m8(__riscv_vand_vx_i16m8(vx_int, 0x3ff, vl), (15 << 10), vl);
        vy = __riscv_vreinterpret_v_i16m8_f16m8(vx_int);

        vy = __riscv_vfdiv_vv_f16m8(__riscv_vfsub_vf_f16m8(vy, 1.41421356237309f, vl), __riscv_vfadd_vf_f16m8(vy, 1.41421356237309f, vl), vl);

        vx = __riscv_vfmul_vv_f16m8(vy, vy, vl);
        v1 = __riscv_vfmul_vf_f16m8(vx, 1.0 / 7, vl);
        v1 = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(v1, 1.0 / 5, vl), vl);
        v1 = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(v1, 1.0 / 3, vl), vl);
        v1 = __riscv_vfmul_vv_f16m8(vy, __riscv_vfadd_vf_f16m8(v1, 1.0, vl), vl);
        vx = __riscv_vfmul_vf_f16m8(v1, 2.0, vl);

        vy = __riscv_vfadd_vv_f16m8(vx, vz, vl);
        // e ^ (alnx)
        vx = __riscv_vfmul_vf_f16m8(vy, exponent, vl);
        vx = __riscv_vfmul_vf_f16m8(vx, 1.4426950408889634f, vl); // log2(e)
        vx_int = __riscv_vfcvt_rtz_x_f_v_i16m8(vx, vl);
        vx = __riscv_vfsub_vv_f16m8(vx, __riscv_vfcvt_f_x_v_f16m8(vx_int, vl), vl);

        vx_int = __riscv_vadd_vx_i16m8(vx_int, 15, vl);
        vx_int = __riscv_vmul_vx_i16m8(vx_int, (1 << 10), vl);
        vy = __riscv_vreinterpret_v_i16m8_f16m8(vx_int);

        vx = __riscv_vfmul_vf_f16m8(vx, 0.693147180559945f, vl);                        // ln2
        vz = __riscv_vfmul_vf_f16m8(vx, 1.0 / 5040, vl);                                // 1/7!
        vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 720, vl), vl); // 1/6!
        vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 120, vl), vl); // 1/5!
        vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 24, vl), vl);  // 1/4!
        vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 6, vl), vl);   // 1/3!
        vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 2, vl), vl);   // 1/2!
        vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0, vl), vl);       // 1/1!
        vz = __riscv_vfadd_vf_f16m8(vz, 1, vl);
        vy = __riscv_vfmul_vv_f16m8(vy, vz, vl);
        __riscv_vse16_v_f16m8(py, vy, vl);
        py += vl;
    }
}

void Pow_float32(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    float32_t exponent = pdat->exponent.v_float32;

    for (size_t i = 0, l = y->ndata; i < l; i++)
        py[i] = pow(px[i], exponent);
}

// x^a = e ^ (alnx)
void Pow_float32_rvv(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    float32_t exponent = pdat->exponent.v_float32;

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t vl;
    vfloat32m8_t vx, vy, vz, v1;
    vint32m8_t vx_int, vk;
    for (; (vl = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= vl) {
        // lnx
        vx = __riscv_vle32_v_f32m8(px, vl);
        px += vl;
        vx_int = __riscv_vreinterpret_v_f32m8_i32m8(vx);
        vk = __riscv_vsra_vx_i32m8(__riscv_vsub_vx_i32m8(vx_int, (127 << 23), vl), 23, vl);

        vz = __riscv_vfadd_vf_f32m8(__riscv_vfcvt_f_x_v_f32m8(vk, vl), 0.5f, vl);
        vz = __riscv_vfmul_vf_f32m8(vz, 0.6931471805599453f, vl); // ln2

        vx_int = __riscv_vor_vx_i32m8(__riscv_vand_vx_i32m8(vx_int, 0x7fffff, vl), (127 << 23), vl);
        vy = __riscv_vreinterpret_v_i32m8_f32m8(vx_int);

        vy = __riscv_vfdiv_vv_f32m8(__riscv_vfsub_vf_f32m8(vy, 1.41421356237309f, vl), __riscv_vfadd_vf_f32m8(vy, 1.41421356237309f, vl), vl);

        vx = __riscv_vfmul_vv_f32m8(vy, vy, vl);
        v1 = __riscv_vfmul_vf_f32m8(vx, 1.0 / 7, vl);
        v1 = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(v1, 1.0 / 5, vl), vl);
        v1 = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(v1, 1.0 / 3, vl), vl);
        v1 = __riscv_vfmul_vv_f32m8(vy, __riscv_vfadd_vf_f32m8(v1, 1.0, vl), vl);
        vx = __riscv_vfmul_vf_f32m8(v1, 2.0, vl);

        vy = __riscv_vfadd_vv_f32m8(vx, vz, vl);
        // e ^ (alnx)
        vx = __riscv_vfmul_vf_f32m8(vy, exponent, vl);
        vx = __riscv_vfmul_vf_f32m8(vx, 1.4426950408889634f, vl); // log2(e)
        vx_int = __riscv_vfcvt_rtz_x_f_v_i32m8(vx, vl);
        vx = __riscv_vfsub_vv_f32m8(vx, __riscv_vfcvt_f_x_v_f32m8(vx_int, vl), vl);

        vx_int = __riscv_vadd_vx_i32m8(vx_int, 127, vl);
        vx_int = __riscv_vmul_vx_i32m8(vx_int, (1 << 23), vl);
        vy = __riscv_vreinterpret_v_i32m8_f32m8(vx_int);

        vx = __riscv_vfmul_vf_f32m8(vx, 0.693147180559945f, vl);                        // ln2
        vz = __riscv_vfmul_vf_f32m8(vx, 1.0 / 5040, vl);                                // 1/7!
        vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1.0 / 720, vl), vl); // 1/6!
        vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1.0 / 120, vl), vl); // 1/5!
        vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1.0 / 24, vl), vl);  // 1/4!
        vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1.0 / 6, vl), vl);   // 1/3!
        vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1.0 / 2, vl), vl);   // 1/2!
        vz = __riscv_vfmul_vv_f32m8(vx, __riscv_vfadd_vf_f32m8(vz, 1.0, vl), vl);       // 1/1!
        vz = __riscv_vfadd_vf_f32m8(vz, 1, vl);
        vy = __riscv_vfmul_vv_f32m8(vy, vz, vl);
        __riscv_vse32_v_f32m8(py, vy, vl);
        py += vl;
    }
}

void *GeneratePowParam(OnnxScalar exponent)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)MALLOC_ASSERT(sizeof(struct operator_pdata_t));
    pdat->exponent = exponent;
    return pdat;
}

void FreePowParam(void **pdat)
{
    free(*pdat);
    *pdat = NULL;
}
