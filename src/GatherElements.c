/**
 * @file GatherElements.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief references:
 *        https://onnx.ai/onnx/operators/onnx__ScatterElements.html
 *        https://pytorch.org/docs/stable/generated/torch.gather.html#torch-gather
 * @version 0.1
 * @date 2024-07-11
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "operators.h"

/**
 * example0:
 * input:
 *       x = [1, 2]
 *           [3, 4]
 * indices = [0, 0]
 *           [1, 0]
 *    axis = 1
 * output:
 *       y = [1, 1]
 *           [4, 3]
 *
 * example1:
 * input:
 *       x = [1, 2, 3]
 *           [4, 5, 6]
 *           [7, 8, 9]
 * indices = [1, 2, 0]
 *           [2, 0, 0]
 *    axis = 0
 * output:
 *       y = [4, 8, 3]
 *           [7, 2, 3]
 *
 */
void GatherElements_int8(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *indices = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    uint8_t *pidx = (uint8_t *)indices->datas;
    int axis = *((int *)node->priv);
    int i, j, idx;

    // assert(x->ndim == indices->ndim);
    // axis == 0 || axis == 1
    // indices->dims[i] == y->dims[i]

    if (axis == 0) {
        for (i = 0; i < indices->dims[1]; ++i) {
            for (j = 0; j < indices->dims[0]; ++j) {
                idx = pidx[i * indices->dims[0] + j];
                py[i * y->dims[0] + j] = px[idx * x->dims[0] + j];
            }
        }
    } else if (axis == 1) {
        for (i = 0; i < indices->dims[1]; ++i) {
            for (j = 0; j < indices->dims[0]; ++j) {
                idx = pidx[i * indices->dims[0] + j];
                py[i * y->dims[0] + j] = px[i * x->dims[1] + idx];
            }
        }
    }
}

void GatherElements_int8_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *indices = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    int8_t *px = (int8_t *)x->datas;
    int8_t *py = (int8_t *)y->datas;
    uint8_t *pidx = (uint8_t *)indices->datas;
    int8_t *pyy;
    uint8_t *pii;
    int axis = *((int *)node->priv);
    vuint8m4_t vidx_load;
    vuint16m8_t vidx;
    vint8m4_t vx;
    size_t avl, vl;

    if (axis == 0) {
        // load row index from indices
        for (int i = 0; i < indices->dims[1]; ++i) {
            avl = indices->dims[0];
            pii = pidx + i * indices->dims[0];
            pyy = py + i * y->dims[0];
            // NOTE: the index need 16bit
            for (; (vl = __riscv_vsetvl_e8m4(avl)) > 0; avl -= vl) {
                // index 0, 1, 2, ...
                vidx = __riscv_vid_v_u16m8(vl);
                // row index row0, row1, row2, ...
                vidx_load = __riscv_vle8_v_u8m4(pii, vl);
                pii += vl;
                // actual index: rown * W + 0, rown * W + 1, rown * W + 2, ...
                vidx = __riscv_vwmaccu_vx_u16m8(vidx, x->dims[0], vidx_load, vl);
                // load data
                vx = __riscv_vluxei16_v_i8m4(px, vidx, vl);
                // store data
                __riscv_vse8_v_i8m4(pyy, vx, vl);
                pyy += vl;
            }
        }
    } else if (axis == 1) {
        // load column index from indices
        for (int i = 0; i < indices->dims[1]; ++i) {
            avl = indices->dims[0];
            pii = pidx + i * indices->dims[0];
            pyy = py + i * y->dims[0];
            for (; (vl = __riscv_vsetvl_e8m4(avl)) > 0; avl -= vl) {
                // column index col0, col1, col2, ...
                vidx_load = __riscv_vle8_v_u8m4(pii, vl);
                pii += vl;
                // convert u8 to u16
                vidx = __riscv_vwcvtu_x_x_v_u16m8(vidx_load, vl);
                // actual index: W * i + coln, W * i + coln + 1, W * i + coln + 2, ...
                vidx = __riscv_vadd_vx_u16m8(vidx, i * x->dims[0], vl);
                // load data
                vx = __riscv_vluxei16_v_i8m4(px, vidx, vl);
                // store data
                __riscv_vse8_v_i8m4(pyy, vx, vl);
                pyy += vl;
            }
        }
    }
}

void GatherElements_int32(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *indices = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    uint8_t *pidx = (uint8_t *)indices->datas;
    int axis = *((int *)node->priv);
    int i, j, idx;

    // assert(x->ndim == indices->ndim);
    // axis == 0 || axis == 1
    // indices->dims[i] == y->dims[i]

    if (axis == 0) {
        for (i = 0; i < indices->dims[1]; ++i) {
            for (j = 0; j < indices->dims[0]; ++j) {
                idx = pidx[i * indices->dims[0] + j];
                py[i * y->dims[0] + j] = px[idx * x->dims[0] + j];
            }
        }
    } else if (axis == 1) {
        for (i = 0; i < indices->dims[1]; ++i) {
            for (j = 0; j < indices->dims[0]; ++j) {
                idx = pidx[i * indices->dims[0] + j];
                py[i * y->dims[0] + j] = px[i * x->dims[1] + idx];
            }
        }
    }
}

void GatherElements_int32_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *indices = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    int32_t *px = (int32_t *)x->datas;
    int32_t *py = (int32_t *)y->datas;
    uint8_t *pidx = (uint8_t *)indices->datas;
    int32_t *pyy;
    uint8_t *pii;
    int axis = *((int *)node->priv);
    vuint8m2_t vidx_load;
    vuint16m4_t vidx;
    vint32m8_t vx;
    size_t avl, vl;

    if (axis == 0) {
        // load row index from indices
        for (int i = 0; i < indices->dims[1]; ++i) {
            avl = indices->dims[0];
            pii = pidx + i * indices->dims[0];
            pyy = py + i * y->dims[0];
            for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                // index 0, 1, 2, ...
                vidx = __riscv_vid_v_u16m4(vl);
                // row index row0, row1, row2, ...
                vidx_load = __riscv_vle8_v_u8m2(pii, vl);
                pii += vl;
                // actual index: rown * W + 0, rown * W + 1, rown * W + 2, ...
                vidx = __riscv_vwmaccu_vx_u16m4(vidx, x->dims[0], vidx_load, vl);
                vidx = __riscv_vmul_vx_u16m4(vidx, sizeof(int32_t), vl);
                // load data
                vx = __riscv_vluxei16_v_i32m8(px, vidx, vl);
                // store data
                __riscv_vse32_v_i32m8(pyy, vx, vl);
                pyy += vl;
            }
        }
    } else if (axis == 1) {
        // load column index from indices
        for (int i = 0; i < indices->dims[1]; ++i) {
            avl = indices->dims[0];
            pii = pidx + i * indices->dims[0];
            pyy = py + i * y->dims[0];
            for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                // column index col0, col1, col2, ...
                vidx_load = __riscv_vle8_v_u8m2(pii, vl);
                pii += vl;
                // convert u8 to u16
                vidx = __riscv_vwcvtu_x_x_v_u16m4(vidx_load, vl);
                // actual index: W * i + coln, W * i + coln + 1, W * i + coln + 2, ...
                vidx = __riscv_vadd_vx_u16m4(vidx, i * x->dims[0], vl);
                vidx = __riscv_vmul_vx_u16m4(vidx, sizeof(int32_t), vl);
                // load data
                vx = __riscv_vluxei16_v_i32m8(px, vidx, vl);
                // store data
                __riscv_vse32_v_i32m8(pyy, vx, vl);
                pyy += vl;
            }
        }
    }
}

void GatherElements_float16(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *indices = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    uint8_t *pidx = (uint8_t *)indices->datas;
    int axis = *((int *)node->priv);
    int i, j, idx;

    // assert(x->ndim == indices->ndim);
    // axis == 0 || axis == 1
    // indices->dims[i] == y->dims[i]

    if (axis == 0) {
        for (i = 0; i < indices->dims[1]; ++i) {
            for (j = 0; j < indices->dims[0]; ++j) {
                idx = pidx[i * indices->dims[0] + j];
                py[i * y->dims[0] + j] = px[idx * x->dims[0] + j];
            }
        }
    } else if (axis == 1) {
        for (i = 0; i < indices->dims[1]; ++i) {
            for (j = 0; j < indices->dims[0]; ++j) {
                idx = pidx[i * indices->dims[0] + j];
                py[i * y->dims[0] + j] = px[i * x->dims[1] + idx];
            }
        }
    }
}

void GatherElements_float16_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *indices = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    float16_t *px = (float16_t *)x->datas;
    float16_t *py = (float16_t *)y->datas;
    uint8_t *pidx = (uint8_t *)indices->datas;
    float16_t *pyy;
    uint8_t *pii;
    int axis = *((int *)node->priv);
    vuint8m4_t vidx_load;
    vuint16m8_t vidx;
    vfloat16m8_t vx;
    size_t avl, vl;

    if (axis == 0) {
        // load row index from indices
        for (int i = 0; i < indices->dims[1]; ++i) {
            avl = indices->dims[0];
            pii = pidx + i * indices->dims[0];
            pyy = py + i * y->dims[0];
            for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
                // index 0, 1, 2, ...
                vidx = __riscv_vid_v_u16m8(vl);
                // row index row0, row1, row2, ...
                vidx_load = __riscv_vle8_v_u8m4(pii, vl);
                pii += vl;
                // actual index: rown * W + 0, rown * W + 1, rown * W + 2, ...
                vidx = __riscv_vwmaccu_vx_u16m8(vidx, x->dims[0], vidx_load, vl);
                vidx = __riscv_vmul_vx_u16m8(vidx, sizeof(float16_t), vl);
                // load data
                vx = __riscv_vluxei16_v_f16m8(px, vidx, vl);
                // store data
                __riscv_vse16_v_f16m8(pyy, vx, vl);
                pyy += vl;
            }
        }
    } else if (axis == 1) {
        // load column index from indices
        for (int i = 0; i < indices->dims[1]; ++i) {
            avl = indices->dims[0];
            pii = pidx + i * indices->dims[0];
            pyy = py + i * y->dims[0];
            for (; (vl = __riscv_vsetvl_e16m8(avl)) > 0; avl -= vl) {
                // column index col0, col1, col2, ...
                vidx_load = __riscv_vle8_v_u8m4(pii, vl);
                pii += vl;
                // convert u8 to u16
                vidx = __riscv_vwcvtu_x_x_v_u16m8(vidx_load, vl);
                // actual index: W * i + coln, W * i + coln + 1, W * i + coln + 2, ...
                vidx = __riscv_vadd_vx_u16m8(vidx, i * x->dims[0], vl);
                vidx = __riscv_vmul_vx_u16m8(vidx, sizeof(float16_t), vl);
                // load data
                vx = __riscv_vluxei16_v_f16m8(px, vidx, vl);
                // store data
                __riscv_vse16_v_f16m8(pyy, vx, vl);
                pyy += vl;
            }
        }
    }
}

void GatherElements_float32(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *indices = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    uint8_t *pidx = (uint8_t *)indices->datas;
    int axis = *((int *)node->priv);
    int i, j, idx;

    // assert(x->ndim == indices->ndim);
    // axis == 0 || axis == 1
    // indices->dims[i] == y->dims[i]

    if (axis == 0) {
        for (i = 0; i < indices->dims[1]; ++i) {
            for (j = 0; j < indices->dims[0]; ++j) {
                idx = pidx[i * indices->dims[0] + j];
                py[i * y->dims[0] + j] = px[idx * x->dims[0] + j];
            }
        }
    } else if (axis == 1) {
        for (i = 0; i < indices->dims[1]; ++i) {
            for (j = 0; j < indices->dims[0]; ++j) {
                idx = pidx[i * indices->dims[0] + j];
                py[i * y->dims[0] + j] = px[i * x->dims[1] + idx];
            }
        }
    }
}

void GatherElements_float32_rvv(struct onnx_node_t *node)
{
    struct onnx_tensor_t *x = node->inputs[0];
    struct onnx_tensor_t *indices = node->inputs[1];
    struct onnx_tensor_t *y = node->outputs[0];
    float32_t *px = (float32_t *)x->datas;
    float32_t *py = (float32_t *)y->datas;
    uint8_t *pidx = (uint8_t *)indices->datas;
    float32_t *pyy;
    uint8_t *pii;
    int axis = *((int *)node->priv);
    vuint8m2_t vidx_load;
    vuint16m4_t vidx;
    vfloat32m8_t vx;
    size_t avl, vl;

    if (axis == 0) {
        // load row index from indices
        for (int i = 0; i < indices->dims[1]; ++i) {
            avl = indices->dims[0];
            pii = pidx + i * indices->dims[0];
            pyy = py + i * y->dims[0];
            for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                // index 0, 1, 2, ...
                vidx = __riscv_vid_v_u16m4(vl);
                // row index row0, row1, row2, ...
                vidx_load = __riscv_vle8_v_u8m2(pii, vl);
                pii += vl;
                // actual index: rown * W + 0, rown * W + 1, rown * W + 2, ...
                vidx = __riscv_vwmaccu_vx_u16m4(vidx, x->dims[0], vidx_load, vl);
                vidx = __riscv_vmul_vx_u16m4(vidx, sizeof(float32_t), vl);
                // load data
                vx = __riscv_vluxei16_v_f32m8(px, vidx, vl);
                // store data
                __riscv_vse32_v_f32m8(pyy, vx, vl);
                pyy += vl;
            }
        }
    } else if (axis == 1) {
        // load column index from indices
        for (int i = 0; i < indices->dims[1]; ++i) {
            avl = indices->dims[0];
            pii = pidx + i * indices->dims[0];
            pyy = py + i * y->dims[0];
            for (; (vl = __riscv_vsetvl_e32m8(avl)) > 0; avl -= vl) {
                // column index col0, col1, col2, ...
                vidx_load = __riscv_vle8_v_u8m2(pii, vl);
                pii += vl;
                // convert u8 to u16
                vidx = __riscv_vwcvtu_x_x_v_u16m4(vidx_load, vl);
                // actual index: W * i + coln, W * i + coln + 1, W * i + coln + 2, ...
                vidx = __riscv_vadd_vx_u16m4(vidx, i * x->dims[0], vl);
                vidx = __riscv_vmul_vx_u16m4(vidx, sizeof(float32_t), vl);
                // load data
                vx = __riscv_vluxei16_v_f32m8(px, vidx, vl);
                // store data
                __riscv_vse32_v_f32m8(pyy, vx, vl);
                pyy += vl;
            }
        }
    }
}