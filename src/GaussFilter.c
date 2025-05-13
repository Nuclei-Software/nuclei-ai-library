/*
 * https://onnx.ai/onnx/operators/onnx__Exp.html#exp
 * https://github.com/shin-mashita/uonnx/blob/main/src/ops/Exp.c
 */

#include "operators.h"
#include "utils.h"

struct operator_pdata_t {
    float sigma;
};

void Gauss_filter_float16(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    float16_t *py = (float16_t *)y->datas;
    float16_t *pa = (float16_t *)a->datas;
    float16_t *pb = (float16_t *)b->datas;
    float16_t sigma = (float16_t)(pdat->sigma);

    float16_t z = 1.0 / (2 * PI * sigma * sigma);
    float16_t inv = -1.0 / (2 * sigma * sigma);

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        py[i] = z * expf((float32_t)((pa[i] * pa[i] + pb[i] * pb[i]) * inv));
    }
}

void Gauss_filter_float16_rvv(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    float16_t *py = (float16_t *)y->datas;
    float16_t *pa = (float16_t *)a->datas;
    float16_t *pb = (float16_t *)b->datas;
    float16_t sigma = (float16_t)(pdat->sigma);

    float16_t z = 1.0 / (2 * PI * sigma * sigma);
    float16_t inv = -1.0 / (2 * sigma * sigma);

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t vl;
    vfloat16m8_t vx, vy, vz;
    vint16m8_t vx_int;
    for (; (vl = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= vl) {
        vx = __riscv_vle16_v_f16m8(pa, vl);
        pa += vl;
        vx = __riscv_vfmul_vv_f16m8(vx, vx, vl);

        vy = __riscv_vle16_v_f16m8(pb, vl);
        pb += vl;
        vy = __riscv_vfmul_vv_f16m8(vy, vy, vl);
    
        vx = __riscv_vfadd_vv_f16m8(vx, vy, vl);
        vx = __riscv_vfmul_vf_f16m8(vx, inv, vl);

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
        vy = __riscv_vfmul_vf_f16m8(vy, z, vl);
        __riscv_vse16_v_f16m8(py, vy, vl);
        py += vl;
    }
}

void Gauss_filter_float32(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    float32_t *py = (float32_t *)y->datas;
    float32_t *pa = (float32_t *)a->datas;
    float32_t *pb = (float32_t *)b->datas;
    float32_t sigma = pdat->sigma;

    float32_t z = 1.0 / (2 * PI * sigma * sigma);
    float32_t inv = -1.0 / (2 * sigma * sigma);

    for (size_t i = 0, l = y->ndata; i < l; i++) {
        py[i] = z * expf((pa[i] * pa[i] + pb[i] * pb[i]) * inv);
    }
}

void Gauss_filter_float32_rvv(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    float32_t *py = (float32_t *)y->datas;
    float32_t *pa = (float32_t *)a->datas;
    float32_t *pb = (float32_t *)b->datas;
    float32_t sigma = pdat->sigma;

    float32_t z = 1.0 / (2 * PI * sigma * sigma);
    float32_t inv = -1.0 / (2 * sigma * sigma);

    size_t blkCnt = y->ndata; /* Loop counter */
    size_t vl;
    vfloat32m8_t vx, vy, vz;
    vint32m8_t vx_int;
    for (; (vl = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= vl) {
        vx = __riscv_vle32_v_f32m8(pa, vl);
        pa += vl;
        vx = __riscv_vfmul_vv_f32m8(vx, vx, vl);

        vy = __riscv_vle32_v_f32m8(pb, vl);
        pb += vl;
        vy = __riscv_vfmul_vv_f32m8(vy, vy, vl);
    
        vx = __riscv_vfadd_vv_f32m8(vx, vy, vl);
        vx = __riscv_vfmul_vf_f32m8(vx, inv, vl);

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

        vy = __riscv_vfmul_vf_f32m8(vy, z, vl);

        __riscv_vse32_v_f32m8(py, vy, vl);
        py += vl;
    }
}

void *GenerateGaussParam(float32_t sigma)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)MALLOC_ASSERT(sizeof(struct operator_pdata_t));
    pdat->sigma = sigma;
    return pdat;
}

void FreeGaussParam(void **pdat)
{
    free(*pdat);
    *pdat = NULL;
}
