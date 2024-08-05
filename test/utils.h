/**
 * @file utils.h
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __UTILS_H__
#define __UTILS_H__

#include <string.h>
#include "onnx.h"
#include "operators.h"
#include "nmsis_bench.h"

int verify_results_int8(int8_t *ref, int8_t *opt, int length);
int verify_results_int32(int32_t *ref, int32_t *opt, int length);
int verify_results_f16(float16_t *ref, float16_t *opt, int length);
int verify_results_f32(float32_t *ref, float32_t *opt, int length);

void show_tensor_int8(struct onnx_tensor_t *t, const char *name);
void show_tensor_bool(struct onnx_tensor_t *t, const char *name);
void show_tensor_f16(struct onnx_tensor_t *t, const char *name);
void show_tensor_f32(struct onnx_tensor_t *t, const char *name);
void show_tensor_int32(struct onnx_tensor_t *t, const char *name);

void HeapSort(int32_t *a, int32_t n);

static inline int csrr_vlenb()
{
    int a = 0;
    asm volatile("csrr %0, vlenb" : "=r"(a) : : "memory");
    return a;
}

#endif