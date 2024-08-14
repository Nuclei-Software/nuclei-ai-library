#ifndef __ONNX_H__
#define __ONNX_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <riscv_vector.h>

typedef float float32_t;
typedef _Float16 float16_t;
typedef double float64_t;

#define PI (3.14159265358979f)

#define MIN(a, b) ({typeof(a) _amin = (a); typeof(b) _bmin = (b); (void)(&_amin == &_bmin); _amin < _bmin ? _amin : _bmin;})
#define MAX(a, b) ({typeof(a) _amax = (a); typeof(b) _bmax = (b); (void)(&_amax == &_bmax); _amax > _bmax ? _amax : _bmax;})

enum onnx_tensor_type_t {
    ONNX_TENSOR_TYPE_UNDEFINED = 0,
    ONNX_TENSOR_TYPE_BOOL = 9,
    ONNX_TENSOR_TYPE_INT8 = 3,
    ONNX_TENSOR_TYPE_INT16 = 5,
    ONNX_TENSOR_TYPE_INT32 = 6,
    ONNX_TENSOR_TYPE_INT64 = 7,
    ONNX_TENSOR_TYPE_UINT8 = 2,
    ONNX_TENSOR_TYPE_UINT16 = 4,
    ONNX_TENSOR_TYPE_UINT32 = 12,
    ONNX_TENSOR_TYPE_UINT64 = 13,
    ONNX_TENSOR_TYPE_BFLOAT16 = 16,
    ONNX_TENSOR_TYPE_FLOAT16 = 10,
    ONNX_TENSOR_TYPE_FLOAT32 = 1,
    ONNX_TENSOR_TYPE_FLOAT64 = 11,
    ONNX_TENSOR_TYPE_COMPLEX64 = 14,
    ONNX_TENSOR_TYPE_COMPLEX128 = 15,
    ONNX_TENSOR_TYPE_STRING = 8,
};

struct onnx_tensor_t {
    char *name;
    enum onnx_tensor_type_t type;
    int *strides;
    int *dims;
    int ndim;
    void *datas;
    size_t ndata;
};

struct onnx_node_t {
    struct onnx_tensor_t **inputs;
    int ninput;
    struct onnx_tensor_t **outputs;
    int noutput;
    void *priv; // private data
};

// struct onnx_tensor_t *onnx_tensor_alloc(enum onnx_tensor_type_t type, int *dims, int ndim);
// void onnx_tensor_free(struct onnx_tensor_t *t);
// void onnx_tensor_reinit(struct onnx_tensor_t *t, enum onnx_tensor_type_t type, int *dims, int ndim);

#ifdef __cplusplus
}
#endif

#endif /* __ONNX_H__ */
