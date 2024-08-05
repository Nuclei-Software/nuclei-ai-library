/**
 * @file MatMul.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

/*
 * https://onnx.ai/onnx/operators/onnx__MatMul.html#matmul
 * https://github.com/xboot/libonnx/blob/master/src/default/MatMul.c
 */

#include "operators.h"

// Following numpy.matmul for shape inference:
// https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html
// TODO(jdqiu): implement MatMul as onnxruntime
// assert(a->ndim == 2 && b->ndim == 2 && a->dims[1] == b->dims[0]);

void MatMul_int8(struct onnx_node_t *n)
{
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    int8_t *py = (int8_t *)y->datas;
    int8_t *pa = (int8_t *)a->datas;
    int8_t *pb = (int8_t *)b->datas;
    int32_t sum;

    for (int i = 0; i < a->dims[1]; ++i) {
        for (int j = 0; j < b->dims[0]; ++j) {
            sum = 0;
            for (int k = 0; k < a->dims[0]; ++k) {
                sum += pa[i * a->dims[0] + k] * pb[k * b->dims[0] + j];
            }
            py[i * b->dims[0] + j] = sum;
        }
    }
}

void MatMul_int8_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    int8_t *py = (int8_t *)y->datas;
    int8_t *pa = (int8_t *)a->datas;
    int8_t *pb = (int8_t *)b->datas;
    uint32_t numColsB = b->dims[0]; /* number of columns of input matrix B */
    uint32_t numColsA = a->dims[0]; /* number of columns of input matrix A */
    uint32_t numRowsA = a->dims[1]; /* number of rows of input matrix A    */
    uint32_t numRowsB = b->dims[1]; /* Number of rows of input matrix B */
    uint32_t colCnt;

    size_t ii, jj, kk;
    size_t l;
    vint8m1_t va0m1, va1m1, va2m1, va3m1;
    vint8m2_t va0m2, va1m2;
    vint32m4_t vres0m4, vres1m4, vres2m4, vres3m4;
    vint32m8_t vres0m8, vres1m8;
    int8_t *px = NULL;
    int8_t *pInA = pa;
    int8_t *pInB = pb;

    colCnt = numRowsA;

    /* ch = 4, mul = 4 */
    for (jj = colCnt / 4; jj > 0; jj--) {
        px = py;
        pInB = pb;
        for (ii = numColsB; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e32m4(ii);
            pInA = pa;
            vres0m4 = __riscv_vmv_v_x_i32m4(0, l);
            vres1m4 = __riscv_vmv_v_v_i32m4(vres0m4, l);
            vres2m4 = __riscv_vmv_v_v_i32m4(vres0m4, l);
            vres3m4 = __riscv_vmv_v_v_i32m4(vres0m4, l);
            for (kk = 0; kk < numColsA; kk++) {
                va0m1 = __riscv_vle8_v_i8m1(pInB + kk * numColsB, l);
                vres0m4 = __riscv_vwmacc_vx_i32m4(vres0m4, *(pInA), __riscv_vwadd_vx_i16m2(va0m1, 0, l), l);
                vres1m4 = __riscv_vwmacc_vx_i32m4(vres1m4, *(pInA + numColsA), __riscv_vwadd_vx_i16m2(va0m1, 0, l), l);
                vres2m4 = __riscv_vwmacc_vx_i32m4(vres2m4, *(pInA + 2 * numColsA), __riscv_vwadd_vx_i16m2(va0m1, 0, l), l);
                vres3m4 = __riscv_vwmacc_vx_i32m4(vres3m4, *(pInA + 3 * numColsA), __riscv_vwadd_vx_i16m2(va0m1, 0, l), l);
                pInA++;
            }
            va0m1 = __riscv_vnsra_wx_i8m1(__riscv_vnsra_wx_i16m2(vres0m4, 0, l), 0, l);
            va1m1 = __riscv_vnsra_wx_i8m1(__riscv_vnsra_wx_i16m2(vres1m4, 0, l), 0, l);
            va2m1 = __riscv_vnsra_wx_i8m1(__riscv_vnsra_wx_i16m2(vres2m4, 0, l), 0, l);
            va3m1 = __riscv_vnsra_wx_i8m1(__riscv_vnsra_wx_i16m2(vres3m4, 0, l), 0, l);
            __riscv_vse8_v_i8m1(px, va0m1, l);
            __riscv_vse8_v_i8m1(px + numColsB, va1m1, l);
            __riscv_vse8_v_i8m1(px + 2 * numColsB, va2m1, l);
            __riscv_vse8_v_i8m1(px + 3 * numColsB, va3m1, l);
            px += l;
            pInB += l;
        }
        pa += 4 * numColsA;
        py += 4 * numColsB;
    }

    /* ch = 2, mul = 8 */
    colCnt = colCnt & 0x3;
    for (jj = colCnt / 2; jj > 0; jj--) {
        px = py;
        pInB = pb;
        for (ii = numColsB; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e32m8(ii);
            pInA = pa;
            vres0m8 = __riscv_vmv_v_x_i32m8(0, l);
            vres1m8 = __riscv_vmv_v_v_i32m8(vres0m8, l);
            for (kk = 0; kk < numColsA; kk++) {
                va0m2 = __riscv_vle8_v_i8m2(pInB + kk * numColsB, l);
                vres0m8 = __riscv_vwmacc_vx_i32m8(vres0m8, *(pInA), __riscv_vwadd_vx_i16m4(va0m2, 0, l), l);
                vres1m8 = __riscv_vwmacc_vx_i32m8(vres1m8, *(pInA + numColsA), __riscv_vwadd_vx_i16m4(va0m2, 0, l), l);
                pInA++;
            }
            va0m2 = __riscv_vnsra_wx_i8m2(__riscv_vnsra_wx_i16m4(vres0m8, 0, l), 0, l);
            va1m2 = __riscv_vnsra_wx_i8m2(__riscv_vnsra_wx_i16m4(vres1m8, 0, l), 0, l);
            __riscv_vse8_v_i8m2(px, va0m2, l);
            __riscv_vse8_v_i8m2(px + numColsB, va1m2, l);
            px += l;
            pInB += l;
        }
        pa += 2 * numColsA;
        py += 2 * numColsB;
    }
    /* ch = 1, mul = 8 */
    colCnt = colCnt & 0x1;
    for (jj = colCnt; jj > 0; jj--) {
        px = py;
        pInB = pb;
        for (ii = numColsB; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e32m8(ii);
            pInA = pa;
            vres0m8 = __riscv_vmv_v_x_i32m8(0, l);
            for (kk = 0; kk < numColsA; kk++) {
                va0m2 = __riscv_vle8_v_i8m2(pInB + kk * numColsB, l);
                vres0m8 = __riscv_vwmacc_vx_i32m8(vres0m8, *(pInA++), __riscv_vwadd_vx_i16m4(va0m2, 0, l), l);
            }
            va0m2 = __riscv_vnsra_wx_i8m2(__riscv_vnsra_wx_i16m4(vres0m8, 0, l), 0, l);
            __riscv_vse8_v_i8m2(px, va0m2, l);
            px += l;
            pInB += l;
        }
        pa += numColsA;
        py += numColsB;
    }
}

void MatMul_float16(struct onnx_node_t *n)
{
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    float16_t *py = (float16_t *)y->datas;
    float16_t *pa = (float16_t *)a->datas;
    float16_t *pb = (float16_t *)b->datas;
    float16_t sum;

    for (int i = 0; i < a->dims[1]; ++i) {
        for (int j = 0; j < b->dims[0]; ++j) {
            sum = 0;
            for (int k = 0; k < a->dims[0]; ++k) {
                sum += pa[i * a->dims[0] + k] * pb[k * b->dims[0] + j];
            }
            py[i * b->dims[0] + j] = sum;
        }
    }
}

void MatMul_float16_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    float16_t *py = (float16_t *)y->datas;
    float16_t *pa = (float16_t *)a->datas;
    float16_t *pb = (float16_t *)b->datas;
    uint32_t numColsB = b->dims[0]; /* number of columns of input matrix B */
    uint32_t numColsA = a->dims[0]; /* number of columns of input matrix A */
    uint32_t numRowsA = a->dims[1]; /* number of rows of input matrix A    */
    uint32_t numRowsB = b->dims[1]; /* Number of rows of input matrix B */
    uint32_t colCnt;

    size_t ii, jj, kk;
    size_t l;
    vfloat16m4_t va0m4, vres0m4, vres1m4, vres2m4, vres3m4;
    vfloat16m8_t va0m8, vres0m8, vres1m8;

    float16_t *px = NULL;
    float16_t *pInA = pa;
    float16_t *pInB = pb;

    colCnt = numRowsA;

    /* ch = 4, mul = 4 */
    for (jj = colCnt / 4; jj > 0; jj--) {
        px = py;
        pInB = pb;
        for (ii = numColsB; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e16m4(ii);
            pInA = pa;
            vres0m4 = __riscv_vfmv_v_f_f16m4(0.0, l);
            vres1m4 = __riscv_vmv_v_v_f16m4(vres0m4, l);
            vres2m4 = __riscv_vmv_v_v_f16m4(vres0m4, l);
            vres3m4 = __riscv_vmv_v_v_f16m4(vres0m4, l);
            for (kk = 0; kk < numColsA; kk++) {
                va0m4 = __riscv_vle16_v_f16m4(pInB + kk * numColsB, l);
                vres0m4 = __riscv_vfmacc_vf_f16m4(vres0m4, *(pInA), va0m4, l);
                vres1m4 = __riscv_vfmacc_vf_f16m4(vres1m4, *(pInA + numColsA), va0m4, l);
                vres2m4 = __riscv_vfmacc_vf_f16m4(vres2m4, *(pInA + 2 * numColsA), va0m4, l);
                vres3m4 = __riscv_vfmacc_vf_f16m4(vres3m4, *(pInA + 3 * numColsA), va0m4, l);
                pInA++;
            }
            __riscv_vse16_v_f16m4(px, vres0m4, l);
            __riscv_vse16_v_f16m4(px + numColsB, vres1m4, l);
            __riscv_vse16_v_f16m4(px + 2 * numColsB, vres2m4, l);
            __riscv_vse16_v_f16m4(px + 3 * numColsB, vres3m4, l);
            px += l;
            pInB += l;
        }
        pa += 4 * numColsA;
        py += 4 * numColsB;
    }
    /* ch = 2, mul = 8 */
    colCnt = colCnt & 0x3;
    for (jj = colCnt / 2; jj > 0; jj--) {
        px = py;
        pInB = pb;
        for (ii = numColsB; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e16m8(ii);
            pInA = pa;
            vres0m8 = __riscv_vfmv_v_f_f16m8(0.0, l);
            vres1m8 = __riscv_vmv_v_v_f16m8(vres0m8, l);
            for (kk = 0; kk < numColsA; kk++) {
                va0m8 = __riscv_vle16_v_f16m8(pInB + kk * numColsB, l);
                vres0m8 = __riscv_vfmacc_vf_f16m8(vres0m8, *(pInA), va0m8, l);
                vres1m8 = __riscv_vfmacc_vf_f16m8(vres1m8, *(pInA + numColsA), va0m8, l);
                pInA++;
            }
            __riscv_vse16_v_f16m8(px, vres0m8, l);
            __riscv_vse16_v_f16m8(px + numColsB, vres1m8, l);
            px += l;
            pInB += l;
        }
        pa += 2 * numColsA;
        py += 2 * numColsB;
    }
    /* ch = 1, mul = 8 */
    colCnt = colCnt & 0x1;
    for (jj = colCnt; jj > 0; jj--) {
        px = py;
        pInB = pb;
        for (ii = numColsB; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e16m8(ii);
            pInA = pa;
            vres0m8 = __riscv_vfmv_v_f_f16m8(0.0, l);
            for (kk = 0; kk < numColsA; kk++) {
                va0m8 = __riscv_vle16_v_f16m8(pInB + kk * numColsB, l);
                vres0m8 = __riscv_vfmacc_vf_f16m8(vres0m8, *(pInA++), va0m8, l);
            }
            __riscv_vse16_v_f16m8(px, vres0m8, l);
            px += l;
            pInB += l;
        }
        pa += numColsA;
        py += numColsB;
    }
}

void MatMul_float32(struct onnx_node_t *n)
{
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    float32_t *py = (float32_t *)y->datas;
    float32_t *pa = (float32_t *)a->datas;
    float32_t *pb = (float32_t *)b->datas;
    float32_t sum;

    for (int i = 0; i < a->dims[1]; ++i) {
        for (int j = 0; j < b->dims[0]; ++j) {
            sum = 0;
            for (int k = 0; k < a->dims[0]; ++k) {
                sum += pa[i * a->dims[0] + k] * pb[k * b->dims[0] + j];
            }
            py[i * b->dims[0] + j] = sum;
        }
    }
}

void MatMul_float32_rvv(struct onnx_node_t *n)
{
    struct onnx_tensor_t *y = n->outputs[0];
    struct onnx_tensor_t *a = n->inputs[0];
    struct onnx_tensor_t *b = n->inputs[1];
    float32_t *py = (float32_t *)y->datas;
    float32_t *pa = (float32_t *)a->datas;
    float32_t *pb = (float32_t *)b->datas;
    uint32_t numColsB = b->dims[0]; /* number of columns of input matrix B */
    uint32_t numColsA = a->dims[0]; /* number of columns of input matrix A */
    uint32_t numRowsA = a->dims[1]; /* number of rows of input matrix A    */
    uint32_t numRowsB = b->dims[1]; /* Number of rows of input matrix B */
    uint32_t colCnt;

    size_t ii, jj, kk;
    size_t l;
    vfloat32m4_t va0m4, vres0m4, vres1m4, vres2m4, vres3m4;
    vfloat32m8_t va0m8, vres0m8, vres1m8;
    colCnt = numRowsA;
    float32_t *px = NULL;
    float32_t *pInA = pa;
    float32_t *pInB = pb;

    /* ch = 4, mul = 4 */
    for (jj = colCnt / 4; jj > 0; jj--) {
        px = py;
        pInB = pb;
        for (ii = numColsB; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e32m4(ii);
            pInA = pa;
            vres0m4 = __riscv_vfmv_v_f_f32m4(0.0, l);
            vres1m4 = __riscv_vmv_v_v_f32m4(vres0m4, l);
            vres2m4 = __riscv_vmv_v_v_f32m4(vres0m4, l);
            vres3m4 = __riscv_vmv_v_v_f32m4(vres0m4, l);
            for (kk = 0; kk < numColsA; kk++) {
                va0m4 = __riscv_vle32_v_f32m4(pInB + kk * numColsB, l);
                vres0m4 = __riscv_vfmacc_vf_f32m4(vres0m4, *(pInA + 0), va0m4, l);
                vres1m4 = __riscv_vfmacc_vf_f32m4(vres1m4, *(pInA + numColsA), va0m4, l);
                vres2m4 = __riscv_vfmacc_vf_f32m4(vres2m4, *(pInA + 2 * numColsA), va0m4, l);
                vres3m4 = __riscv_vfmacc_vf_f32m4(vres3m4, *(pInA + 3 * numColsA), va0m4, l);
                pInA++;
            }
            __riscv_vse32_v_f32m4(px, vres0m4, l);
            __riscv_vse32_v_f32m4(px + numColsB, vres1m4, l);
            __riscv_vse32_v_f32m4(px + 2 * numColsB, vres2m4, l);
            __riscv_vse32_v_f32m4(px + 3 * numColsB, vres3m4, l);
            px += l;
            pInB += l;
        }
        pa += 4 * numColsA;
        py += 4 * numColsB;
    }
    /* ch = 2, mul = 8 */
    colCnt = colCnt & 0x3;
    for (jj = colCnt / 2; jj > 0; jj--) {
        px = py;
        pInB = pb;
        for (ii = numColsB; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e32m8(ii);
            pInA = pa;
            vres0m8 = __riscv_vfmv_v_f_f32m8(0.0, l);
            vres1m8 = __riscv_vmv_v_v_f32m8(vres0m8, l);
            for (kk = 0; kk < numColsA; kk++) {
                va0m8 = __riscv_vle32_v_f32m8(pInB + kk * numColsB, l);
                vres0m8 = __riscv_vfmacc_vf_f32m8(vres0m8, *(pInA + 0), va0m8, l);
                vres1m8 = __riscv_vfmacc_vf_f32m8(vres1m8, *(pInA + numColsA), va0m8, l);
                pInA++;
            }
            __riscv_vse32_v_f32m8(px, vres0m8, l);
            __riscv_vse32_v_f32m8(px + numColsB, vres1m8, l);
            px += l;
            pInB += l;
        }
        pa += 2 * numColsA;
        py += 2 * numColsB;
    }
    /* ch = 1, mul = 8 */
    colCnt = colCnt & 0x1;
    for (jj = colCnt; jj > 0; jj--) {
        px = py;
        pInB = pb;
        for (ii = numColsB; ii > 0; ii -= l) {
            l = __riscv_vsetvl_e32m8(ii);
            pInA = pa;
            vres0m8 = __riscv_vfmv_v_f_f32m8(0.0, l);
            for (kk = 0; kk < numColsA; kk++) {
                va0m8 = __riscv_vle32_v_f32m8(pInB + kk * numColsB, l);
                vres0m8 = __riscv_vfmacc_vf_f32m8(vres0m8, *(pInA++), va0m8, l);
            }
            __riscv_vse32_v_f32m8(px, vres0m8, l);
            px += l;
            pInB += l;
        }
        pa += numColsA;
        py += numColsB;
    }
}