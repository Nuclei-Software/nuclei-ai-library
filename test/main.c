/**
 * @file main.c
 * @author qiujiandong (qiujiandong@nucleisys.com)
 * @brief
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "utils.h"

typedef struct {
    int (*func)(void);
    const char *name;
} TestFunc;

extern int test_abs(void);
extern int test_add(void);
extern int test_batchnormalization(void);
extern int test_clamp(void);
extern int test_concat(void);
extern int test_cos(void);
extern int test_div(void);
extern int test_elu(void);
extern int test_exp(void);
extern int test_flip(void);
extern int test_gatherelements(void);
extern int test_layernormalization(void);
extern int test_log(void);
extern int test_matmul(void);
extern int test_mul(void);
extern int test_negate(void);
extern int test_pad(void);
extern int test_pow(void);
extern int test_reciprocal(void);
extern int test_reduce(void);
extern int test_Relu(void);
extern int test_rmsnormalization(void);
extern int test_rsqrt(void);
extern int test_scatterelements(void);
extern int test_silu(void);
extern int test_sin(void);
extern int test_slice(void);
extern int test_softmax(void);
extern int test_sqrt(void);
extern int test_sub(void);
extern int test_tile(void);
extern int test_topk(void);

TestFunc tests[] = {
    {test_abs, "test_abs"},
    {test_add, "test_add"},
    {test_batchnormalization, "test_batchnormalization"},
    {test_clamp, "test_clamp"},
    {test_concat, "test_concat"},
    {test_cos, "test_cos"},
    {test_div, "test_div"},
    {test_elu, "test_elu"},
    {test_exp, "test_exp"},
    {test_flip, "test_flip"},
    {test_gatherelements, "test_gatherelements"},
    {test_layernormalization, "test_layernormalization"},
    {test_log, "test_log"},
    {test_matmul, "test_matmul"},
    {test_mul, "test_mul"},
    {test_negate, "test_negate"},
    {test_pad, "test_pad"},
    {test_pow, "test_pow"},
    {test_reciprocal, "test_reciprocal"},
    {test_reduce, "test_reduce"},
    {test_Relu, "test_Relu"},
    {test_rmsnormalization, "test_rmsnormalization"},
    {test_rsqrt, "test_rsqrt"},
    {test_scatterelements, "test_scatterelements"},
    {test_silu, "test_silu"},
    {test_sin, "test_sin"},
    {test_slice, "test_slice"},
    {test_softmax, "test_softmax"},
    {test_sqrt, "test_sqrt"},
    {test_sub, "test_sub"},
    {test_tile, "test_tile"},
    {test_topk, "test_topk"},
};

int main(void)
{
    int has_failed = 0;
    int results[sizeof(tests) / sizeof(tests[0])];
#ifndef __riscv_vector
#error "Not support this cpu arch, need v ext!!"
#endif
    printf("\r\nvlen = %d bits\r\n\r\n", csrr_vlenb() * 8); // vlen = vlenb * 8

#ifdef CSR_BF16
    __RV_CSR_SET(CSR_MFP16MODE, 0x1);
    printf("BF16 csr(07E2) = 0x%x\r\n", __RV_CSR_READ(CSR_MFP16MODE));
#endif

#ifdef VLM_LATENCY
// Note: config vlm latency, data should put in vlm
#define MISC_BASE_ADDR 0x10012000
#define VLM_OFFSET 0x28
    uintptr_t latency_reg = MISC_BASE_ADDR + VLM_OFFSET; // reg addr
    printf("vlm latency cfg before = %d\r\n", *(uintptr_t *)latency_reg);
    *(uintptr_t *)latency_reg = 9; // 0-19
    printf("vlm latency cfg after = %d\r\n", *(uintptr_t *)latency_reg);
#endif

    for (int i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        results[i] = tests[i].func();
    }

    printf("All test done!\n-------------\n");

    for (int i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        if (results[i] != 0) {
            printf("Test %s failed!\n", tests[i].name);
            has_failed = 1;
        }
    }
    if (!has_failed) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("Some tests failed!\n");
        return -1;
    }
}
