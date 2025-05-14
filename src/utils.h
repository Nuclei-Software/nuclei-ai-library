#ifndef __UTILS_H__
#define __UTILS_H__

#include "nmsis_bench.h"
#include "onnx.h"
#include "operators.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int verify_results_int8(int8_t *ref, int8_t *opt, int length);
int verify_results_uint8(uint8_t *ref, uint8_t *opt, int length);
int verify_results_int32(int32_t *ref, int32_t *opt, int length);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
int verify_results_f16(float16_t *ref, float16_t *opt, int length);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
int verify_results_bf16(bfloat16_t *ref, bfloat16_t *opt, int length);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */

int verify_results_f32(float32_t *ref, float32_t *opt, int length);

void show_tensor_int8(struct onnx_tensor_t *t, const char *name);
void show_tensor_bool(struct onnx_tensor_t *t, const char *name);
void show_tensor_f16(struct onnx_tensor_t *t, const char *name);
void show_tensor_f32(struct onnx_tensor_t *t, const char *name);
void show_tensor_int32(struct onnx_tensor_t *t, const char *name);

void HeapSort_int32(int32_t *a, int32_t n);
#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void HeapSort_f16(float16_t *a, int32_t n);
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */
#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void HeapSort_bf16(bfloat16_t *a, int32_t n);
#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */
void HeapSort_f32(float32_t *a, int32_t n);

#define MALLOC_ASSERT(size) malloc_assert(size, __FILE__, __LINE__)

static inline void *malloc_assert(size_t size, const char *file, int line)
{
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed in %s at line %d\n", file, line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#endif
