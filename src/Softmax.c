/*
 * https://onnx.ai/onnx/operators/onnx__Softmax.html#softmax
 * https://github.com/shin-mashita/uonnx/blob/main/src/ops/Softmax.c
 */

#include "operators.h"

// TODO(jdqiu): add argument "axis" to control N and D
// default axis = 1 in opset < 13 or axis = -1 in opset = 13
// assert(x->ndim == 2);

void Softmax_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    float16_t maxv, sum, v;
    int i, j, o;

    const int D = x->dims[0];
    const int N = x->dims[1];

    for (i = 0, o = 0; i < N; i++, o += D) {
        for (j = 0, maxv = px[o]; j < D; j++) {
            if (px[o + j] > maxv)
                maxv = px[o + j];
        }
        for (j = 0, sum = 0; j < D; j++) {
            py[o + j] = expf(px[o + j] - maxv);
            sum += py[o + j];
        }
        if (sum != 0) {
            float16_t inv = 1.0 / sum;
            for (j = 0; j < D; j++)
                py[o + j] *= inv;
        }
    }
}

void Softmax_float16_rvv(struct onnx_node_t *n)
{
    struct operator_softmax_1_11_pdata_t *pdat = (struct operator_softmax_1_11_pdata_t *)n->priv;
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    float16_t maxv, sum, v;
    int i, j, o;
    const int D = x->dims[0];
    const int N = x->dims[1];

    for (i = 0, o = 0; i < N; i++, o += D) {
        size_t blkCnt = D; /* Loop counter */
        size_t l;
        float16_t maxValue = px[o];
        float16_t *pSrc = px + o;
        vfloat16m8_t v_in;

        l = __riscv_vsetvl_e16m1(1);
        vfloat16m1_t v_max = __riscv_vfmv_s_f_f16m1(maxValue, l);
        for (; (l = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= l) {
            v_in = __riscv_vle16_v_f16m8(pSrc, l);
            pSrc += l;
            v_max = __riscv_vfredmax_vs_f16m8_f16m1(v_in, v_max, l);
        }
        maxv = __riscv_vfmv_f_s_f16m1_f16(v_max);

        vfloat16m8_t vx, vy, vz;
        vint16m8_t vx_int;
        blkCnt = D;
        pSrc = px + o;
        float16_t *z = py + o;

        l = __riscv_vsetvl_e16m1(1);
        vfloat16m1_t vsum = __riscv_vfsub_vv_f16m1(vsum, vsum, l);
        for (; (l = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= l) {
            vx = __riscv_vle16_v_f16m8(pSrc, l);
            pSrc += l;
            vx = __riscv_vfsub_vf_f16m8(vx, maxv, l);

            vx = __riscv_vfmul_vf_f16m8(vx, 1.4426950408889634f, l); // log2(e)

            vx_int = __riscv_vfcvt_rtz_x_f_v_i16m8(vx, l);
            vx = __riscv_vfsub_vv_f16m8(vx, __riscv_vfcvt_f_x_v_f16m8(vx_int, l), l);
            vx_int = __riscv_vadd_vx_i16m8(vx_int, 15, l);
            vx_int = __riscv_vmul_vx_i16m8(vx_int, (1 << 10), l);
            vy = __riscv_vreinterpret_v_i16m8_f16m8(vx_int);
            vx = __riscv_vfmul_vf_f16m8(vx, 0.693147180559945f, l);                       // ln2
            vz = __riscv_vfmul_vf_f16m8(vx, 1.0 / 5040, l);                               // 1/7!
            vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 720, l), l); // 1/6!
            vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 120, l), l); // 1/5!
            vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 24, l), l);  // 1/4!
            vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 6, l), l);   // 1/3!
            vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0 / 2, l), l);   // 1/2!
            vz = __riscv_vfmul_vv_f16m8(vx, __riscv_vfadd_vf_f16m8(vz, 1.0, l), l);       // 1/1!
            vz = __riscv_vfadd_vf_f16m8(vz, 1, l);
            vy = __riscv_vfmul_vv_f16m8(vy, vz, l);

            vsum = __riscv_vfredusum_vs_f16m8_f16m1(vy, vsum, l);

            __riscv_vse16_v_f16m8(z, vy, l);
            z += l;
        }
        sum = __riscv_vfmv_f_s_f16m1_f16(vsum);

        if (sum == 0)
            return;

        float16_t inv = 1.0 / sum;
        blkCnt = D;
        z = py + o;
        for (; (l = __riscv_vsetvl_e16m8(blkCnt)) > 0; blkCnt -= l) {
            vx = __riscv_vle16_v_f16m8(z, l);
            vy = __riscv_vfmul_vf_f16m8(vx, inv, l);
            __riscv_vse16_v_f16m8(z, vy, l);
            z += l;
        }
    }
}

void Softmax_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    float32_t maxv, sum;
    int i, j, o;

    const int D = x->dims[0];
    const int N = x->dims[1];

    for (i = 0, o = 0; i < N; i++, o += D) {
        for (j = 0, maxv = px[o]; j < D; j++) {
            if (px[o + j] > maxv)
                maxv = px[o + j];
        }
        for (j = 0, sum = 0; j < D; j++) {
            py[o + j] = expf(px[o + j] - maxv);
            sum += py[o + j];
        }

        if (sum != 0) {
            float32_t inv = 1.0 / sum;
            for (j = 0; j < D; j++)
                py[o + j] *= inv;
        }
    }
}

void Softmax_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *x = n->inputs[0];
    struct onnx_tensor_t *y = n->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    float32_t maxv, sum;
    int i, j, o;

    const int D = x->dims[0];
    const int N = x->dims[1];

    for (i = 0, o = 0; i < N; i++, o += D) {
        size_t blkCnt = D; /* Loop counter */
        size_t vl;
        float32_t maxValue = px[o];
        float32_t *pSrc = px + o;
        vfloat32m8_t v_in;

        vl = __riscv_vsetvl_e32m1(1);
        vfloat32m1_t v_max = __riscv_vfmv_s_f_f32m1(maxValue, vl);
        for (; (vl = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= vl) {
            v_in = __riscv_vle32_v_f32m8(pSrc, vl);
            pSrc += vl;
            v_max = __riscv_vfredmax_vs_f32m8_f32m1(v_in, v_max, vl);
        }
        maxv = __riscv_vfmv_f_s_f32m1_f32(v_max);

        vfloat32m8_t vx, vy, vz;
        vint32m8_t vx_int;
        blkCnt = D;
        pSrc = px + o;
        float32_t *z = py + o;

        vl = __riscv_vsetvl_e32m1(1);
        vfloat32m1_t vsum = __riscv_vfsub_vv_f32m1(vsum, vsum, vl);
        for (; (vl = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= vl) {
            vx = __riscv_vle32_v_f32m8(pSrc, vl);
            pSrc += vl;
            vx = __riscv_vfsub_vf_f32m8(vx, maxv, vl);

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

            vsum = __riscv_vfredusum_vs_f32m8_f32m1(vy, vsum, vl);

            __riscv_vse32_v_f32m8(z, vy, vl);
            z += vl;
        }
        sum = __riscv_vfmv_f_s_f32m1_f32(vsum);

        if (sum == 0)
            return;

        float32_t inv = 1.0 / sum;
        blkCnt = D;
        z = py + o;
        for (; (vl = __riscv_vsetvl_e32m8(blkCnt)) > 0; blkCnt -= vl) {
            vx = __riscv_vle32_v_f32m8(z, vl);
            vy = __riscv_vfmul_vf_f32m8(vx, inv, vl);
            __riscv_vse32_v_f32m8(z, vy, vl);
            z += vl;
        }
    }
}
