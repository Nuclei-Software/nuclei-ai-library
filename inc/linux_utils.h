#ifndef __LINUX_UTILS__
#define __LINUX_UTILS__

#ifdef __cplusplus
 extern "C" {
#endif

#include <stdio.h>
#include <stdint.h>


#define  __ALIGNED(x) __attribute__((aligned(x)))
#define __STATIC_FORCEINLINE static inline __attribute__((always_inline))
#define __STATIC_INLINE static inline
#define __WEAK

#define __ASM_STR(x)    #x

#define csr_read(csr)                                           \
({                                                              \
        register unsigned long __v;                             \
        __asm__ __volatile__ ("csrr %0, " __ASM_STR(csr)        \
                              : "=r" (__v) :                    \
                              : "memory");                      \
        __v;                                                    \
})

#define __RV_CSR_SET(csr, val)                                  \
 ({                                                             \
        register unsigned long __v = (unsigned long )val;       \
        __asm__ volatile("csrs " __ASM_STR(csr) ", %0"          \
                     :                                          \
                     : "rK"(__v)                                \
                     : "memory");                               \
})

#define __RV_CSR_CLEAR(csr, val)                                \
 ({                                                             \
        register unsigned long __v = (unsigned long )val;       \
        __asm__ volatile("csrc " __ASM_STR(csr) ", %0"          \
                     :                                          \
                     : "rK"(__v)                                \
                     : "memory");                               \
})

#ifndef CSR_MFP16MODE
#define CSR_MFP16MODE 0x7E2
#endif

#ifdef __cplusplus
}
#endif
#endif /* __LINUX_UTILS__ */
