#include <stdbool.h>

#include "utils.h"

#define DELTAF32 (0.1f)
#define DELTABF16 (1)
#define DELTAINT8 (1)
#define DELTAINT32 (1)

void csr_set_bf16_mode(void)
{
    __RV_CSR_SET(CSR_MFP16MODE, 0x1);
    //printf("BF16 csr(07E2) = 0x%x\r\n", __RV_CSR_READ(CSR_MFP16MODE));
}

void csr_clr_bf16_mode(void)
{
    __RV_CSR_CLEAR(CSR_MFP16MODE, 0x1);
}

int verify_results_int8(int8_t *ref, int8_t *opt, int length)
{
    int flag = 0;

    for (int i = 0; i < length; i++) {
        if (abs((int)(ref[i] - opt[i])) > DELTAINT8) {
            printf("INT8 Output mismatch at %d, expected %d, actual %d\r\n", i, ref[i], opt[i]);
            flag = 1;
            break;
        }
    }

    return flag;
}

int verify_results_uint8(uint8_t *ref, uint8_t *opt, int length)
{
    int flag = 0;

    for (int i = 0; i < length; i++) {
        if (abs(ref[i] - opt[i]) > DELTAINT8) {
            printf("UINT8 Output mismatch at %d, expected %u, actual %u\r\n", i, ref[i], opt[i]);
            flag = 1;
            break;
        }
    }

    return flag;
}

int verify_results_int32(int32_t *ref, int32_t *opt, int length)
{
    int8_t flag = 0;

    for (int i = 0; i < length; i++) {
        if (abs(ref[i] - opt[i]) > DELTAINT32) {
            printf("INT32 Output mismatch at %d, expected %d, actual %d\r\n", i, ref[i], opt[i]);
            flag = 1;
            break;
        }
    }

    return flag;
}

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)

int verify_results_f16(float16_t *ref, float16_t *opt, int length)
{

    int8_t flag = 0;
    float32_t f32_ref, f32_opt;

    for (int i = 0; i < length; i++) {
        f32_ref = (float32_t)ref[i];
        f32_opt = (float32_t)opt[i];
        if (fabs(f32_ref - f32_opt) > DELTAF32) {
            printf("F16 Output mismatch at %d, expected %f, actual %f\r\n", i, f32_ref, f32_opt);
            flag = 1;
            break;
        }
    }

    return flag;
}

#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */

#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)

int verify_results_bf16(bfloat16_t *ref, bfloat16_t *opt, int length)
{

    int8_t flag = 0;
    float32_t f32_ref, f32_opt;

    for (int i = 0; i < length; i++) {
        f32_ref = (float32_t)ref[i];
        f32_opt = (float32_t)opt[i];
        if (fabs(f32_ref - f32_opt) > DELTABF16) {
            printf("BF16 Output mismatch at %d, expected %f, actual %f\r\n", i, f32_ref, f32_opt);
            flag = 1;
            break;
        }
    }

    return flag;
}

#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */

int verify_results_f32(float32_t *ref, float32_t *opt, int length)
{
    int8_t flag = 0;

    for (int i = 0; i < length; i++) {
        if (fabs(ref[i] - opt[i]) > DELTAF32) {
            printf("F32 Output mismatch at %d, expected %f, actual %f\r\n", i, ref[i], opt[i]);
            flag = 1;
            break;
        }
        // printf("f32 Output mismatch at %d, expected %f, actual %f\r\n", i, ref[i], opt[i]);
    }

    return flag;
}

void show_tensor_int8_impl(struct onnx_tensor_t *t, size_t offset, int dim)
{
    printf("[ ");
    if (dim == 0) {
        for (int i = 0; i < t->dims[dim]; i++) {
            printf("%d ", ((int8_t *)t->datas)[offset + i]);
        }
    } else {
        for (int i = 0; i < t->dims[dim]; i++) {
            show_tensor_int8_impl(t, offset + i * t->strides[dim], dim - 1);
        }
    }
    printf("]\n");
}

void show_tensor_int8(struct onnx_tensor_t *t, const char *name)
{
    printf("%s: \n", name);
    show_tensor_int8_impl(t, 0, t->ndim - 1);
}

void show_tensor_bool_impl(struct onnx_tensor_t *t, size_t offset, int dim)
{
    printf("[ ");
    if (dim == 0) {
        for (int i = 0; i < t->dims[dim]; i++) {
            printf("%d ", ((bool *)t->datas)[offset + i]);
        }
    } else {
        for (int i = 0; i < t->dims[dim]; i++) {
            show_tensor_bool_impl(t, offset + i * t->strides[dim], dim - 1);
        }
    }
    printf("]\n");
}

void show_tensor_bool(struct onnx_tensor_t *t, const char *name)
{
    printf("%s: \n", name);
    show_tensor_bool_impl(t, 0, t->ndim - 1);
}

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)
void show_tensor_f16_impl(struct onnx_tensor_t *t, size_t offset, int dim)
{
    float32_t tmp;
    printf("[ ");
    if (dim == 0) {
        for (int i = 0; i < t->dims[dim]; i++) {
            tmp = (float32_t)(((float16_t *)t->datas)[offset + i]);
            printf("%f ", tmp);
        }
    } else {
        for (int i = 0; i < t->dims[dim]; i++) {
            show_tensor_f16_impl(t, offset + i * t->strides[dim], dim - 1);
        }
    }
    printf("]\n");
}

void show_tensor_f16(struct onnx_tensor_t *t, const char *name)
{
    printf("%s: \n", name);
    show_tensor_f16_impl(t, 0, t->ndim - 1);
}
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */

#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)
void show_tensor_bf16_impl(struct onnx_tensor_t *t, size_t offset, int dim)
{
    float32_t tmp;
    printf("[ ");
    if (dim == 0) {
        for (int i = 0; i < t->dims[dim]; i++) {
            tmp = (float32_t)(((bfloat16_t *)t->datas)[offset + i]);
            printf("%f ", tmp);
        }
    } else {
        for (int i = 0; i < t->dims[dim]; i++) {
            show_tensor_bf16_impl(t, offset + i * t->strides[dim], dim - 1);
        }
    }
    printf("]\n");
}

void show_tensor_bf16(struct onnx_tensor_t *t, const char *name)
{
    printf("%s: \n", name);
    show_tensor_bf16_impl(t, 0, t->ndim - 1);
}
#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */

void show_tensor_int32_impl(struct onnx_tensor_t *t, size_t offset, int dim)
{
    printf("[ ");
    if (dim == 0) {
        for (int i = 0; i < t->dims[dim]; i++) {
            printf("%d ", ((int32_t *)t->datas)[offset + i]);
        }
    } else {
        for (int i = 0; i < t->dims[dim]; i++) {
            show_tensor_int32_impl(t, offset + i * t->strides[dim], dim - 1);
        }
    }
    printf("]\n");
}

void show_tensor_int32(struct onnx_tensor_t *t, const char *name)
{
    printf("%s: \n", name);
    show_tensor_int32_impl(t, 0, t->ndim - 1);
}

void show_tensor_f32_impl(struct onnx_tensor_t *t, size_t offset, int dim)
{
    printf("[ ");
    if (dim == 0) {
        for (int i = 0; i < t->dims[dim]; i++) {
            printf("%f ", ((float32_t *)t->datas)[offset + i]);
        }
    } else {
        for (int i = 0; i < t->dims[dim]; i++) {
            show_tensor_f32_impl(t, offset + i * t->strides[dim], dim - 1);
        }
    }
    printf("]\n");
}

void show_tensor_f32(struct onnx_tensor_t *t, const char *name)
{
    printf("%s: \n", name);
    show_tensor_int32_impl(t, 0, t->ndim - 1);
}

static void Swap_int32(int32_t *a, int32_t *b)
{
    int32_t tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

static void Heapify_int32(int32_t arr[], int len, int idx)
{
    int child = idx * 2 + 1;
    while (child < len) {
        if (child + 1 < len && arr[child + 1] < arr[child]) {
            ++child;
        }

        if (arr[child] < arr[idx]) {
            Swap_int32(&arr[child], &arr[idx]);
            idx = child;
            child = idx * 2 + 1;
        } else {
            break;
        }
    }
}

void HeapSort_int32(int32_t *a, int32_t n)
{
    for (int32_t i = (n - 1 - 1) / 2; i >= 0; --i) {
        Heapify_int32(a, n, i);
    }
    for (int32_t j = n; j > 0; --j) {
        Swap_int32(&a[0], &a[j - 1]);
        Heapify_int32(a, j - 1, 0);
    }
}

#if defined(RISCV_FLOAT16_RVV_SUPPORTED)

static void Swap_f16(float16_t *a, float16_t *b)
{
    float16_t tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

static void Heapify_f16(float16_t arr[], int len, int idx)
{
    int child = idx * 2 + 1;
    while (child < len) {
        if (child + 1 < len && arr[child + 1] < arr[child]) {
            ++child;
        }

        if (arr[child] < arr[idx]) {
            Swap_f16(&arr[child], &arr[idx]);
            idx = child;
            child = idx * 2 + 1;
        } else {
            break;
        }
    }
}

void HeapSort_f16(float16_t *a, int32_t n)
{
    for (int32_t i = (n - 1 - 1) / 2; i >= 0; --i) {
        Heapify_f16(a, n, i);
    }
    for (int32_t j = n; j > 0; --j) {
        Swap_f16(&a[0], &a[j - 1]);
        Heapify_f16(a, j - 1, 0);
    }
}

#endif /* #if defined(RISCV_FLOAT16_RVV_SUPPORTED) */

#if defined(RISCV_BFLOAT16_RVV_SUPPORTED)

static void Swap_bf16(bfloat16_t *a, bfloat16_t *b)
{
    bfloat16_t tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

static void Heapify_bf16(bfloat16_t arr[], int len, int idx)
{
    int child = idx * 2 + 1;
    while (child < len) {
        if (child + 1 < len && arr[child + 1] < arr[child]) {
            ++child;
        }

        if (arr[child] < arr[idx]) {
            Swap_bf16(&arr[child], &arr[idx]);
            idx = child;
            child = idx * 2 + 1;
        } else {
            break;
        }
    }
}

void HeapSort_bf16(bfloat16_t *a, int32_t n)
{
    for (int32_t i = (n - 1 - 1) / 2; i >= 0; --i) {
        Heapify_bf16(a, n, i);
    }
    for (int32_t j = n; j > 0; --j) {
        Swap_bf16(&a[0], &a[j - 1]);
        Heapify_bf16(a, j - 1, 0);
    }
}

#endif /* #if defined(RISCV_BFLOAT16_RVV_SUPPORTED) */

static void Swap_f32(float32_t *a, float32_t *b)
{
    float32_t tmp;
    tmp = *a;
    *a = *b;
    *b = tmp;
}

static void Heapify_f32(float32_t arr[], int len, int idx)
{
    int child = idx * 2 + 1;
    while (child < len) {
        if (child + 1 < len && arr[child + 1] < arr[child]) {
            ++child;
        }

        if (arr[child] < arr[idx]) {
            Swap_f32(&arr[child], &arr[idx]);
            idx = child;
            child = idx * 2 + 1;
        } else {
            break;
        }
    }
}

void HeapSort_f32(float32_t *a, int32_t n)
{
    for (int32_t i = (n - 1 - 1) / 2; i >= 0; --i) {
        Heapify_f32(a, n, i);
    }
    for (int32_t j = n; j > 0; --j) {
        Swap_f32(&a[0], &a[j - 1]);
        Heapify_f32(a, j - 1, 0);
    }
}