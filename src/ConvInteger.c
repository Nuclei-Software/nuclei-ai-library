/*
 * https://onnx.ai/onnx/operators/onnx__ConvInteger.html
 */

#include "operators.h"
#include "utils.h"

typedef struct {
    void *buf;
    size_t buf_size;
} Context;

typedef struct {
    int32_t w;
    int32_t h;
} Tile;

typedef struct {
    int32_t min;
    int32_t max;
} Activation;

struct riscv_nn_double {
    uint32_t low;
    int32_t high;
};

union riscv_nn_long_long {
    int64_t long_long;
    struct riscv_nn_double word;
};

struct operator_pdata_t {
    Context ctx;
    int32_t input_offset;  /**< The negative of the zero value for the input tensor */
    int32_t output_offset; /**< The negative of the zero value for the output tensor */
    Tile stride;
    Tile padding;
    Tile dilation;
    Activation activation;
};

#define LEFT_SHIFT(_shift) (_shift > 0 ? _shift : 0)
#define RIGHT_SHIFT(_shift) (_shift > 0 ? 0 : -_shift)

__STATIC_FORCEINLINE int32_t requantize(const int32_t val, const int32_t multiplier, const int32_t shift)
{
    int32_t result = 0;
    union riscv_nn_long_long mult;

    // Rounding offset to add for a right shift of 31
    mult.word.low = 1 << 30;
    mult.word.high = 0;

    // Gets resolved as a SMLAL instruction
    mult.long_long = mult.long_long + (int64_t)(val * (1 << LEFT_SHIFT(shift))) * multiplier;

    // Utilize all of the upper 32 bits. This is the doubling step
    // as well.
    result = (int32_t)(mult.long_long >> 31);

    const int32_t remainder_mask = (1 << RIGHT_SHIFT(shift)) - 1;
    int32_t remainder = remainder_mask & result;

    // Basic division
    result >>= RIGHT_SHIFT(shift);

    // Adjust 'result' for rounding (mid point away from zero)
    int32_t threshold = remainder_mask >> 1;
    if (result < 0) {
        threshold++;
    }
    if (remainder > threshold) {
        result++;
    }

    return result;
}

static int8_t *mat_mult_kernel_row_offset_s8_s16(const int8_t *input_a, const int16_t *input_b, const uint16_t output_ch, const int32_t *out_shift,
                                                 const int32_t *out_mult, const int32_t out_offset, const int16_t activation_min,
                                                 const int16_t activation_max, const int32_t num_col_a, const int32_t aligned_num_col_a,
                                                 const int32_t *const output_bias, const int32_t row_address_offset, int8_t *out_0)
{

    /* set up the second output pointers */

    int8_t *out_1 = out_0 + row_address_offset;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch / 2;
    const int8_t *ip_a0 = input_a;
    /* this loop over rows in A */
    while (row_count) {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + aligned_num_col_a;

        /* align the second pointer for A */
        const int8_t *ip_a1 = ip_a0 + num_col_a;

        int32_t ch_0_out_0 = 0;
        int32_t ch_0_out_1 = 0;
        int32_t ch_1_out_0 = 0;
        int32_t ch_1_out_1 = 0;
        /* Init accumulator with bias for channel N and N + 1 */
        if (bias) {
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias++;
            ch_1_out_0 = *bias;
            ch_1_out_1 = *bias++;
        }

        int32_t col_count = num_col_a;
        while (col_count) {
            int8_t a0 = *ip_a0++;
            int16_t b0 = *ip_b0++;
            int8_t a1 = *ip_a1++;
            int16_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            ch_1_out_0 += a1 * b0;
            ch_1_out_1 += a1 * b1;
            col_count--;
        } /* while over col_count */

        ch_0_out_0 = requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (int8_t)ch_0_out_0;

        ch_0_out_1 = requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (int8_t)ch_0_out_1;
        out_mult++;
        out_shift++;

        ch_1_out_0 = requantize(ch_1_out_0, *out_mult, *out_shift);
        ch_1_out_0 += out_offset;
        ch_1_out_0 = MAX(ch_1_out_0, activation_min);
        ch_1_out_0 = MIN(ch_1_out_0, activation_max);
        *out_0++ = (int8_t)ch_1_out_0;

        ch_1_out_1 = requantize(ch_1_out_1, *out_mult, *out_shift);
        ch_1_out_1 += out_offset;
        ch_1_out_1 = MAX(ch_1_out_1, activation_min);
        ch_1_out_1 = MIN(ch_1_out_1, activation_max);
        *out_1++ = (int8_t)ch_1_out_1;
        out_mult++;
        out_shift++;

        /* skip row */
        ip_a0 += num_col_a;
        row_count--;
    }

    /* compute the last odd numbered row if any */
    if (output_ch & 0x1) {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + aligned_num_col_a;

        int32_t ch_0_out_0 = 0;
        int32_t ch_0_out_1 = 0;

        /* load the bias */
        if (bias) {
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias++;
        }

        int32_t col_count = num_col_a;
        while (col_count) {
            int8_t a0 = *ip_a0++;
            int16_t b0 = *ip_b0++;
            int16_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            col_count--;
        }

        ch_0_out_0 = requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (int8_t)ch_0_out_0;

        ch_0_out_1 = requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (int8_t)ch_0_out_1;
        out_mult++;
        out_shift++;
    }

    out_0 += 2 * row_address_offset - output_ch;

    /* return the new output pointer with offset */
    return out_0;
}

static int8_t *mat_mult_kernel_s8_s16(const int8_t *input_a, const int16_t *input_b, const uint16_t output_ch, const int32_t *out_shift,
                                      const int32_t *out_mult, const int32_t out_offset, const int16_t activation_min, const int16_t activation_max,
                                      const int32_t num_col_a, const int32_t aligned_num_col_a, const int32_t *const output_bias, int8_t *out_0)
{
    /* set up the second output pointers */
    int8_t *out_1 = out_0 + output_ch;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch / 2;
    const int8_t *ip_a0 = input_a;
    /* this loop over rows in A */
    while (row_count) {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + aligned_num_col_a;

        /* align the second pointer for A */
        const int8_t *ip_a1 = ip_a0 + num_col_a;

        int32_t ch_0_out_0 = 0;
        int32_t ch_0_out_1 = 0;
        int32_t ch_1_out_0 = 0;
        int32_t ch_1_out_1 = 0;
        /* Init accumulator with bias for channel N and N + 1 */
        if (bias) {
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias++;
            ch_1_out_0 = *bias;
            ch_1_out_1 = *bias++;
        }

        int32_t col_count = num_col_a;
        while (col_count) {
            int8_t a0 = *ip_a0++;
            int16_t b0 = *ip_b0++;
            int8_t a1 = *ip_a1++;
            int16_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            ch_1_out_0 += a1 * b0;
            ch_1_out_1 += a1 * b1;
            col_count--;
        } /* while over col_count */

        ch_0_out_0 = requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (int8_t)ch_0_out_0;

        ch_0_out_1 = requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (int8_t)ch_0_out_1;
        out_mult++;
        out_shift++;

        ch_1_out_0 = requantize(ch_1_out_0, *out_mult, *out_shift);
        ch_1_out_0 += out_offset;
        ch_1_out_0 = MAX(ch_1_out_0, activation_min);
        ch_1_out_0 = MIN(ch_1_out_0, activation_max);
        *out_0++ = (int8_t)ch_1_out_0;

        ch_1_out_1 = requantize(ch_1_out_1, *out_mult, *out_shift);
        ch_1_out_1 += out_offset;
        ch_1_out_1 = MAX(ch_1_out_1, activation_min);
        ch_1_out_1 = MIN(ch_1_out_1, activation_max);
        *out_1++ = (int8_t)ch_1_out_1;
        out_mult++;
        out_shift++;

        /* skip row */
        ip_a0 += num_col_a;
        row_count--;
    }

    /* compute the last odd numbered row if any */
    if (output_ch & 0x1) {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + aligned_num_col_a;

        int32_t ch_0_out_0 = 0;
        int32_t ch_0_out_1 = 0;

        /* load the bias */
        if (bias) {
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias++;
        }

        int32_t col_count = num_col_a;

        while (col_count) {
            int8_t a0 = *ip_a0++;
            int16_t b0 = *ip_b0++;
            int16_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            col_count--;
        }
        ch_0_out_0 = requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (int8_t)ch_0_out_0;

        ch_0_out_1 = requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (int8_t)ch_0_out_1;
        out_mult++;
        out_shift++;
    }

    out_0 += output_ch;

    /* return the new output pointer with offset */
    return out_0;
}

int ConvInteger(struct onnx_node_t *n)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    struct onnx_tensor_t *input = n->inputs[0];    // shape [batch, input_h, input_w, input_ch]
    struct onnx_tensor_t *filter = n->inputs[1];   // shape [output_ch, kernel_h, kernel_w, input_ch]
    struct onnx_tensor_t *bias = n->inputs[2];     // shape [output_ch]
    struct onnx_tensor_t *multiply = n->inputs[3]; // shape [output_ch]
    struct onnx_tensor_t *shift = n->inputs[4];    // shape [output_ch]

    struct onnx_tensor_t *output = n->outputs[0]; // shape [batch, output_h, output_w, output_ch]
    int16_t *buffer_a = (int16_t *)pdat->ctx.buf;

    const int32_t input_batches = input->dims[3];
    const uint16_t input_y = input->dims[2];
    const uint16_t input_x = input->dims[1];
    const uint16_t input_ch = input->dims[0];

    const uint16_t kernel_x = filter->dims[1];
    const uint16_t kernel_y = filter->dims[2];
    const uint16_t kernel_ch = filter->dims[0];
    // output_ch: filter->dims[3] == output->dims[0]

    const uint16_t output_y = output->dims[2];
    const uint16_t output_x = output->dims[1];
    const uint16_t output_ch = output->dims[0];

    const uint16_t pad_x = pdat->padding.w;
    const uint16_t pad_y = pdat->padding.h;
    const uint16_t stride_x = pdat->stride.w;
    const uint16_t stride_y = pdat->stride.h;
    const int32_t dilation_x = pdat->dilation.w;
    const int32_t dilation_y = pdat->dilation.h;
    const int32_t out_offset = pdat->output_offset;
    const int32_t out_activation_min = pdat->activation.min;
    const int32_t out_activation_max = pdat->activation.max;
    const int32_t input_offset = pdat->input_offset;

    const int32_t groups = input_ch / kernel_ch;
    const int32_t rhs_cols = kernel_x * kernel_y * kernel_ch;
    const int32_t output_ch_per_group = output_ch / groups;

    const int8_t *input_data = input->datas;
    int8_t *output_data = output->datas;

    int32_t *output_mult = multiply->datas;
    int32_t *output_shift = shift->datas;

    if (input_ch % groups != 0 || output_ch % groups != 0) {
        return -1;
    }

    const int32_t remainder = rhs_cols % 4;
    const int32_t aligned_rhs_cols = remainder != 0 ? rhs_cols + 4 - remainder : rhs_cols;

    for (int i_batch = 0; i_batch < input_batches; i_batch++) {

        /* Use as a ping-pong buffer for unordered elements */
        int8_t *im2col_buf = (int8_t *)buffer_a + aligned_rhs_cols * 2;
        int16_t *im2col_buf_start_s16 = buffer_a;
        int32_t lhs_rows = 0;

        const int8_t *filter_data_ptr = filter->datas;
        const int32_t *bias_data_ptr = bias->datas;
        const int32_t *output_mult_ptr = &output_mult[0];
        const int32_t *output_shift_ptr = &output_shift[0];

        /* This part implements the im2col function */
        for (int32_t i_group = 0; i_group < groups; i_group++) {
            int8_t *out = output_data + i_group * output_ch_per_group;
            for (int i_out_y = 0; i_out_y < output_y; i_out_y++) {
                for (int i_out_x = 0; i_out_x < output_x; i_out_x++) {
                    const int32_t base_idx_x = stride_x * i_out_x - pad_x;
                    const int32_t base_idx_y = stride_y * i_out_y - pad_y;

                    for (int32_t i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++) {
                        for (int32_t i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++) {
                            const int32_t k_y = base_idx_y + dilation_y * i_ker_y;
                            const int32_t k_x = base_idx_x + dilation_x * i_ker_x;

                            if (k_y < 0 || k_y >= input_y || k_x < 0 || k_x >= input_x) {
                                memset(im2col_buf, (int8_t)-input_offset, sizeof(int8_t) * kernel_ch);
                            } else {
                                memcpy(im2col_buf, input_data + (k_y * input_x + k_x) * input_ch + i_group * kernel_ch, sizeof(int8_t) * kernel_ch);
                            }
                            im2col_buf += kernel_ch;
                        }
                    }
                    lhs_rows++;

                    int32_t block_cnt = rhs_cols;
                    const int8_t *src = im2col_buf - rhs_cols;
                    int16_t *dst = im2col_buf_start_s16;
                    while (block_cnt > 0) {
                        *dst++ = (int16_t)*src++ + input_offset;
                        block_cnt--;
                    }

                    im2col_buf_start_s16 += aligned_rhs_cols;

                    if (lhs_rows == 2) {
                        if (groups > 1) {
                            out = mat_mult_kernel_row_offset_s8_s16(filter_data_ptr, buffer_a, output_ch_per_group, output_shift_ptr, output_mult_ptr,
                                                                    out_offset, out_activation_min, out_activation_max, rhs_cols, aligned_rhs_cols,
                                                                    bias_data_ptr, output_ch, out);
                        } else {
                            out =
                                mat_mult_kernel_s8_s16(filter_data_ptr, buffer_a, output_ch_per_group, output_shift_ptr, output_mult_ptr, out_offset,
                                                       out_activation_min, out_activation_max, rhs_cols, aligned_rhs_cols, bias_data_ptr, out);
                        }

                        /* counter reset */
                        im2col_buf_start_s16 = buffer_a;
                        im2col_buf = (int8_t *)buffer_a + aligned_rhs_cols * 2;
                        lhs_rows = 0;
                    }
                }
            }

            if (out == NULL) {
                return -1;
            }

            /* Handle left over columns */
            if (lhs_rows != 0) {

                const int8_t *ker_a = filter_data_ptr;
                int i;

                for (i = 0; i < output_ch_per_group; i++) {
                    /* Load the accumulator with bias first */
                    int32_t sum = 0;
                    if (bias_data_ptr) {
                        sum = bias_data_ptr[i];
                    }

                    const int16_t *ip_as_col = buffer_a;
                    uint16_t col_count = rhs_cols;
                    while (col_count) {
                        int8_t ker_a1 = *ker_a++;
                        int16_t ip_b1 = *ip_as_col++;
                        sum += ker_a1 * ip_b1;
                        col_count--;
                    }

                    sum = requantize(sum, output_mult_ptr[i], output_shift_ptr[i]);
                    sum += out_offset;
                    sum = MAX(sum, out_activation_min);
                    sum = MIN(sum, out_activation_max);
                    *out++ = (int8_t)sum;
                }

                im2col_buf_start_s16 = buffer_a;
                im2col_buf = (int8_t *)buffer_a + aligned_rhs_cols * 2;
                lhs_rows = 0;
            }
            filter_data_ptr += output_ch_per_group * rhs_cols;
            bias_data_ptr += output_ch_per_group;
            output_mult_ptr += output_ch_per_group;
            output_shift_ptr += output_ch_per_group;
        }
        /* Advance to the next batch */
        input_data += (input_x * input_y * input_ch);
        output_data += (output_x * output_y * output_ch);
    }

    /* Return to application */
    return 0;
}

static int convolve_3x3_s8_wg23_pad_input(const int *input_dims, int32_t pad_w, int32_t pad_h, int8_t input_offset, const int8_t *input_data,
                                          const Tile *output_shape, int8_t *in_pad)
{
    const int32_t in_w = input_dims[1];
    const int32_t in_h = input_dims[2];
    const int32_t in_ch = input_dims[0];
    const int32_t in_pad_w = output_shape->w + 2;
    const int32_t in_pad_h = output_shape->h + 2;
    const int32_t pad_left = pad_w;
    const int32_t pad_top = pad_h;
    const int8_t pad_val = -input_offset;

    // pad top
    size_t avl, vl;
    vint8m8_t zero = __riscv_vmv_v_x_i8m8(pad_val, __riscv_vsetvlmax_e8m8());
    avl = in_pad_w * in_ch * pad_top;
    for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
        __riscv_vse8_v_i8m8(in_pad, zero, vl);
        in_pad += vl;
    }

    const size_t row_cnt = in_h < in_pad_h - pad_top ? in_h : in_pad_h - pad_top;
    const size_t col_cnt = in_w < in_pad_w - pad_left ? in_w : in_pad_w - pad_left;
    const size_t pad_bottom = in_pad_h - pad_top - row_cnt;
    const size_t pad_right = in_pad_w - pad_left - col_cnt;
    for (int i = 0; i < row_cnt; ++i) {
        // pad left
        avl = pad_left * in_ch;
        for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
            __riscv_vse8_v_i8m8(in_pad, zero, vl);
            in_pad += vl;
        }

        // copy row data
        avl = col_cnt * in_ch;
        vint8m8_t tmp;
        const int8_t *tmp_in = input_data;
        for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
            tmp = __riscv_vle8_v_i8m8(tmp_in, vl);
            tmp_in += vl;
            __riscv_vse8_v_i8m8(in_pad, tmp, vl);
            in_pad += vl;
        }

        input_data += in_w * in_ch; // change to next row

        // pad right
        if (pad_right > 0) {
            avl = pad_right * in_ch;
            for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
                __riscv_vse8_v_i8m8(in_pad, zero, vl);
                in_pad += vl;
            }
        }
    }

    // pad bottom
    if (pad_bottom > 0) {
        avl = in_pad_w * in_ch * pad_bottom;
        for (; (vl = __riscv_vsetvl_e8m8(avl)) > 0; avl -= vl) {
            __riscv_vse8_v_i8m8(in_pad, zero, vl);
            in_pad += vl;
        }
    }

    return 0;
}

static int convolve_3x3_s8_wg23_trans_kernel(const int *kernel_dims, const int8_t *kernel_data, int16_t *kernel_tm)
{
    if (kernel_dims[2] != 3 || kernel_dims[1] != 3) {
        return -1;
    }

    const int32_t out_ch = kernel_dims[3];
    const int32_t in_ch = kernel_dims[0];
    // the shape of kernel_tm should be [N, H, W, C] = [1, C_OUT, C_IN, 16]
    // const int32_t w_step = out_ch * in_ch;
    const ptrdiff_t bstride = 16 * sizeof(int16_t);

    // const float transform_mat[4][3] = {
    //     {1.0f, 0.0f, 0.0f},
    //     {1.0f / 2, 1.0f / 2, 1.0f / 2},
    //     {1.0f / 2, -1.0f / 2, 1.0f / 2},
    //     {0.0f, 0.0f, 1.0f}
    // };

    // result in Q.9 format
    for (int32_t outch_idx = 0; outch_idx < out_ch; outch_idx++) {
        const int8_t *kernel = kernel_data + outch_idx * 9 * in_ch;
        int16_t *kernel_tm_ptr = kernel_tm + outch_idx * in_ch * 16;

        size_t avl = in_ch, vl;
        const int8_t *base = kernel;
        for (; (vl = __riscv_vsetvl_e8mf2(avl)) > 0; avl -= vl) {
            vint8mf2_t k00 = __riscv_vle8_v_i8mf2(base, vl);
            vint8mf2_t k01 = __riscv_vle8_v_i8mf2(base + 1 * in_ch, vl);
            vint8mf2_t k02 = __riscv_vle8_v_i8mf2(base + 2 * in_ch, vl);
            vint8mf2_t k10 = __riscv_vle8_v_i8mf2(base + 3 * in_ch, vl);
            vint8mf2_t k11 = __riscv_vle8_v_i8mf2(base + 4 * in_ch, vl);
            vint8mf2_t k12 = __riscv_vle8_v_i8mf2(base + 5 * in_ch, vl);
            vint8mf2_t k20 = __riscv_vle8_v_i8mf2(base + 6 * in_ch, vl);
            vint8mf2_t k21 = __riscv_vle8_v_i8mf2(base + 7 * in_ch, vl);
            vint8mf2_t k22 = __riscv_vle8_v_i8mf2(base + 8 * in_ch, vl);
            base += vl;

            vint16m1_t sum012 = __riscv_vwadd_wv_i16m1(__riscv_vwadd_vv_i16m1(k00, k01, vl), k02, vl);
            vint16m1_t sum02_1 = __riscv_vwsub_wv_i16m1(__riscv_vwadd_vv_i16m1(k00, k02, vl), k01, vl);
            vint16m1_t sum345 = __riscv_vwadd_wv_i16m1(__riscv_vwadd_vv_i16m1(k10, k11, vl), k12, vl);
            vint16m1_t sum678 = __riscv_vwadd_wv_i16m1(__riscv_vwadd_vv_i16m1(k20, k21, vl), k22, vl);
            vint16m1_t sum68_7 = __riscv_vwsub_wv_i16m1(__riscv_vwadd_vv_i16m1(k20, k22, vl), k21, vl);
            vint16m1_t sum036 = __riscv_vwadd_wv_i16m1(__riscv_vwadd_vv_i16m1(k00, k10, vl), k20, vl);
            vint16m1_t sum06_3 = __riscv_vwsub_wv_i16m1(__riscv_vwadd_vv_i16m1(k00, k20, vl), k10, vl);
            vint16m1_t sum147 = __riscv_vwadd_wv_i16m1(__riscv_vwadd_vv_i16m1(k01, k11, vl), k21, vl);
            vint16m1_t sum17_4 = __riscv_vwsub_wv_i16m1(__riscv_vwadd_vv_i16m1(k01, k21, vl), k11, vl);
            vint16m1_t sum258 = __riscv_vwadd_wv_i16m1(__riscv_vwadd_vv_i16m1(k02, k12, vl), k22, vl);
            vint16m1_t sum28_5 = __riscv_vwsub_wv_i16m1(__riscv_vwadd_vv_i16m1(k02, k22, vl), k12, vl);

            vint16m1x8_t v_tuple;
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 0, __riscv_vsll_vx_i16m1(__riscv_vsext_vf2_i16m1(k00, vl), 2, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 1, __riscv_vsll_vx_i16m1(sum012, 1, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 2, __riscv_vsll_vx_i16m1(sum02_1, 1, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 3, __riscv_vsll_vx_i16m1(__riscv_vsext_vf2_i16m1(k02, vl), 2, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 4, __riscv_vsll_vx_i16m1(sum036, 1, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 5, __riscv_vadd_vv_i16m1(__riscv_vadd_vv_i16m1(sum036, sum258, vl), sum147, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 6, __riscv_vsub_vv_i16m1(__riscv_vadd_vv_i16m1(sum036, sum258, vl), sum147, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 7, __riscv_vsll_vx_i16m1(sum258, 1, vl));
            __riscv_vssseg8e16_v_i16m1x8(kernel_tm_ptr, bstride, v_tuple, vl);

            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 0, __riscv_vsll_vx_i16m1(sum06_3, 1, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 1, __riscv_vsub_vv_i16m1(__riscv_vadd_vv_i16m1(sum012, sum678, vl), sum345, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 2, __riscv_vsub_vv_i16m1(__riscv_vadd_vv_i16m1(sum06_3, sum28_5, vl), sum17_4, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 3, __riscv_vsll_vx_i16m1(sum28_5, 1, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 4, __riscv_vsll_vx_i16m1(__riscv_vsext_vf2_i16m1(k20, vl), 2, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 5, __riscv_vsll_vx_i16m1(sum678, 1, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 6, __riscv_vsll_vx_i16m1(sum68_7, 1, vl));
            v_tuple = __riscv_vset_v_i16m1_i16m1x8(v_tuple, 7, __riscv_vsll_vx_i16m1(__riscv_vsext_vf2_i16m1(k22, vl), 2, vl));
            __riscv_vssseg8e16_v_i16m1x8(kernel_tm_ptr + 8, bstride, v_tuple, vl);

            kernel_tm_ptr += 16 * vl;
        }
    }

    return 0;
}

static void trans_input_rowop(const int8_t *tile_start, int32_t w_step, int32_t c_in, int16_t *result)
{
    const int8_t *tile_in_row0 = tile_start;
    const int8_t *tile_in_row1 = tile_in_row0 + w_step;
    const int8_t *tile_in_row2 = tile_in_row1 + w_step;
    const int8_t *tile_in_row3 = tile_in_row2 + w_step;

    // buffer size is 16 x c_in
    int16_t *tile_row0 = result;
    int16_t *tile_row1 = tile_row0 + c_in * 4;
    int16_t *tile_row2 = tile_row1 + c_in * 4;
    int16_t *tile_row3 = tile_row2 + c_in * 4;

    size_t avl = c_in * 4, vl;
    for (; (vl = __riscv_vsetvl_e8m1(avl)) > 0; avl -= vl) {
        // load row datas
        vint8m1_t row0 = __riscv_vle8_v_i8m1(tile_in_row0, vl);
        tile_in_row0 += vl;
        vint8m1_t row1 = __riscv_vle8_v_i8m1(tile_in_row1, vl);
        tile_in_row1 += vl;
        vint8m1_t row2 = __riscv_vle8_v_i8m1(tile_in_row2, vl);
        tile_in_row2 += vl;
        vint8m1_t row3 = __riscv_vle8_v_i8m1(tile_in_row3, vl);
        tile_in_row3 += vl;

        // left hand matrix multiply with data
        __riscv_vse16_v_i16m2(tile_row0, __riscv_vwsub_vv_i16m2(row0, row2, vl), vl);
        tile_row0 += vl;
        __riscv_vse16_v_i16m2(tile_row1, __riscv_vwadd_vv_i16m2(row1, row2, vl), vl);
        tile_row1 += vl;
        __riscv_vse16_v_i16m2(tile_row2, __riscv_vwsub_vv_i16m2(row2, row1, vl), vl);
        tile_row2 += vl;
        __riscv_vse16_v_i16m2(tile_row3, __riscv_vwsub_vv_i16m2(row1, row3, vl), vl);
        tile_row3 += vl;
    }
}

static void trans_input_col_op(const int16_t *buffer, int32_t c_in, int16_t *result)
{
    const ptrdiff_t bstride = 16 * sizeof(int16_t);
    size_t avl = c_in, vl;
    const int16_t *tile_row0 = buffer;
    const int16_t *tile_row1 = tile_row0 + c_in * 4;
    const int16_t *tile_row2 = tile_row1 + c_in * 4;
    const int16_t *tile_row3 = tile_row2 + c_in * 4;

    for (; (vl = __riscv_vsetvl_e16mf2(avl)) > 0; avl -= vl) {
        vint16mf2x8_t v_tuple;

        vint16mf2_t d00 = __riscv_vle16_v_i16mf2(tile_row0, vl);
        vint16mf2_t d01 = __riscv_vle16_v_i16mf2(tile_row0 + c_in, vl);
        vint16mf2_t d02 = __riscv_vle16_v_i16mf2(tile_row0 + c_in * 2, vl);
        vint16mf2_t d03 = __riscv_vle16_v_i16mf2(tile_row0 + c_in * 3, vl);
        tile_row0 += vl;
        vint16mf2_t d10 = __riscv_vle16_v_i16mf2(tile_row1, vl);
        vint16mf2_t d11 = __riscv_vle16_v_i16mf2(tile_row1 + c_in, vl);
        vint16mf2_t d12 = __riscv_vle16_v_i16mf2(tile_row1 + c_in * 2, vl);
        vint16mf2_t d13 = __riscv_vle16_v_i16mf2(tile_row1 + c_in * 3, vl);
        tile_row1 += vl;

        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 0, __riscv_vsub_vv_i16mf2(d00, d02, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 1, __riscv_vadd_vv_i16mf2(d01, d02, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 2, __riscv_vsub_vv_i16mf2(d02, d01, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 3, __riscv_vsub_vv_i16mf2(d01, d03, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 4, __riscv_vsub_vv_i16mf2(d10, d12, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 5, __riscv_vadd_vv_i16mf2(d11, d12, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 6, __riscv_vsub_vv_i16mf2(d12, d11, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 7, __riscv_vsub_vv_i16mf2(d11, d13, vl));
        __riscv_vssseg8e16_v_i16mf2x8(result, bstride, v_tuple, vl);

        d00 = __riscv_vle16_v_i16mf2(tile_row2, vl);
        d01 = __riscv_vle16_v_i16mf2(tile_row2 + c_in, vl);
        d02 = __riscv_vle16_v_i16mf2(tile_row2 + c_in * 2, vl);
        d03 = __riscv_vle16_v_i16mf2(tile_row2 + c_in * 3, vl);
        tile_row2 += vl;
        d10 = __riscv_vle16_v_i16mf2(tile_row3, vl);
        d11 = __riscv_vle16_v_i16mf2(tile_row3 + c_in, vl);
        d12 = __riscv_vle16_v_i16mf2(tile_row3 + c_in * 2, vl);
        d13 = __riscv_vle16_v_i16mf2(tile_row3 + c_in * 3, vl);
        tile_row3 += vl;

        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 0, __riscv_vsub_vv_i16mf2(d00, d02, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 1, __riscv_vadd_vv_i16mf2(d01, d02, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 2, __riscv_vsub_vv_i16mf2(d02, d01, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 3, __riscv_vsub_vv_i16mf2(d01, d03, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 4, __riscv_vsub_vv_i16mf2(d10, d12, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 5, __riscv_vadd_vv_i16mf2(d11, d12, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 6, __riscv_vsub_vv_i16mf2(d12, d11, vl));
        v_tuple = __riscv_vset_v_i16mf2_i16mf2x8(v_tuple, 7, __riscv_vsub_vv_i16mf2(d11, d13, vl));
        __riscv_vssseg8e16_v_i16mf2x8(result + 8, bstride, v_tuple, vl);

        result += 16 * vl;
    }
}

static int32_t convolve_3x3_s8_wg23_trans_input(const int32_t *input_dims, const int8_t *input_data, int16_t *in_tm)
{
    // transform input [1, H, W, C_IN] to [1, tiles, C_IN, 16]
    // input_dims->n is not used and assumed to be 1
    const int32_t tile_w = (input_dims[1] - 2) >> 1;
    const int32_t tile_h = (input_dims[2] - 2) >> 1;
    const int32_t tiles = tile_w * tile_h;
    const int32_t h_in = input_dims[2];
    const int32_t w_in = input_dims[1];
    const int32_t c_in = input_dims[0];
    const int32_t w_step = w_in * c_in;

    // B^T = [1,  0, -1,  0]
    //       [0,  1,  1,  0]
    //       [0, -1,  1,  0]
    //       [0,  1,  0, -1]

    // loop over each tile
    int16_t tmp[16];
    for (int32_t h_idx = 0; h_idx < tile_h; ++h_idx) {
        for (int32_t w_idx = 0; w_idx < tile_w; ++w_idx) {
            const int32_t tile_idx = h_idx * tile_w + w_idx;
            const int8_t *tile_start = input_data + 2 * h_idx * w_step + 2 * w_idx * c_in;

            // buffer size is 16 x c_in
            int16_t *buffer = in_tm + tiles * 16 * c_in;
            trans_input_rowop(tile_start, w_step, c_in, buffer);

            int16_t *result = in_tm + tile_idx * c_in * 16;
            trans_input_col_op(buffer, c_in, result);
        }
    }

    return 0;
}

static int32_t riscv_convolve_3x3_s8_wg23_dot(const int32_t *in_tm_dims, const int16_t *in_tm, const int32_t *kernel_tm_dims,
                                              const int16_t *kernel_tm, const int *dot_dims, int32_t *dot)
{
    // shape of in_tm is [N, H, W, C] = [1, tiles, C_IN, 16]
    // shape of kernel_tm is [N, H, W, C] = [1, C_OUT, C_IN, 16]
    // shape of dot is [N, H, W, C] = [1, tiles, C_OUT, 16]
    const int32_t in_ch = in_tm_dims[1];
    const int32_t out_ch = kernel_tm_dims[2];
    const int32_t tiles = in_tm_dims[2];

    uint32_t outch_idx[4] = {0};
    uint32_t tile_idx[4] = {0};

    uint32_t out_idx = 0;
    const int32_t ouput_cnt = tiles * out_ch;

    for (; out_idx + 3 < ouput_cnt; out_idx += 4) {
        {
            vuint32m1_t vidx = __riscv_vadd_vx_u32m1(__riscv_vid_v_u32m1(4), out_idx, 4);
            vuint32m1_t outch_vidx = __riscv_vremu_vx_u32m1(vidx, out_ch, 4);
            __riscv_vse32_v_u32m1(outch_idx, outch_vidx, 4);
            vuint32m1_t tile_vidx = __riscv_vdivu_vx_u32m1(vidx, out_ch, 4);
            __riscv_vse32_v_u32m1(tile_idx, tile_vidx, 4);
        }

        vint32m4_t sum0 = __riscv_vmv_v_x_i32m4(0, 16);
        vint32m4_t sum1 = __riscv_vmv_v_x_i32m4(0, 16);
        vint32m4_t sum2 = __riscv_vmv_v_x_i32m4(0, 16);
        vint32m4_t sum3 = __riscv_vmv_v_x_i32m4(0, 16);

        const int16_t *in0 = in_tm + tile_idx[0] * in_ch * 16;
        const int16_t *in1 = in_tm + tile_idx[1] * in_ch * 16;
        const int16_t *in2 = in_tm + tile_idx[2] * in_ch * 16;
        const int16_t *in3 = in_tm + tile_idx[3] * in_ch * 16;

        const int16_t *kernel0 = kernel_tm + outch_idx[0] * in_ch * 16;
        const int16_t *kernel1 = kernel_tm + outch_idx[1] * in_ch * 16;
        const int16_t *kernel2 = kernel_tm + outch_idx[2] * in_ch * 16;
        const int16_t *kernel3 = kernel_tm + outch_idx[3] * in_ch * 16;

        int32_t *dot0 = dot + out_idx * 16;
        int32_t *dot1 = dot0 + 16;
        int32_t *dot2 = dot1 + 16;
        int32_t *dot3 = dot2 + 16;

        for (int32_t cin_idx = 0; cin_idx < in_ch; ++cin_idx) {
            vint16m2_t vin0 = __riscv_vle16_v_i16m2(in0 + cin_idx * 16, 16);
            vint16m2_t vin1 = __riscv_vle16_v_i16m2(in1 + cin_idx * 16, 16);
            vint16m2_t vin2 = __riscv_vle16_v_i16m2(in2 + cin_idx * 16, 16);
            vint16m2_t vin3 = __riscv_vle16_v_i16m2(in3 + cin_idx * 16, 16);

            vint16m2_t vkernel0 = __riscv_vle16_v_i16m2(kernel0 + cin_idx * 16, 16);
            vint16m2_t vkernel1 = __riscv_vle16_v_i16m2(kernel1 + cin_idx * 16, 16);
            vint16m2_t vkernel2 = __riscv_vle16_v_i16m2(kernel2 + cin_idx * 16, 16);
            vint16m2_t vkernel3 = __riscv_vle16_v_i16m2(kernel3 + cin_idx * 16, 16);

            sum0 = __riscv_vwmacc_vv_i32m4(sum0, vin0, vkernel0, 16);
            sum1 = __riscv_vwmacc_vv_i32m4(sum1, vin1, vkernel1, 16);
            sum2 = __riscv_vwmacc_vv_i32m4(sum2, vin2, vkernel2, 16);
            sum3 = __riscv_vwmacc_vv_i32m4(sum3, vin3, vkernel3, 16);
        }

        __riscv_vse32_v_i32m4(dot0, sum0, 16);
        __riscv_vse32_v_i32m4(dot1, sum1, 16);
        __riscv_vse32_v_i32m4(dot2, sum2, 16);
        __riscv_vse32_v_i32m4(dot3, sum3, 16);
    }

    for (; out_idx + 1 < ouput_cnt; out_idx += 2) {
        {
            vuint32m1_t vidx = __riscv_vadd_vx_u32m1(__riscv_vid_v_u32m1(2), out_idx, 2);
            vuint32m1_t outch_vidx = __riscv_vremu_vx_u32m1(vidx, out_ch, 2);
            __riscv_vse32_v_u32m1(outch_idx, outch_vidx, 2);
            vuint32m1_t tile_vidx = __riscv_vdivu_vx_u32m1(vidx, out_ch, 2);
            __riscv_vse32_v_u32m1(tile_idx, tile_vidx, 4);
        }

        vint32m4_t sum0 = __riscv_vmv_v_x_i32m4(0, 16);
        vint32m4_t sum1 = __riscv_vmv_v_x_i32m4(0, 16);

        const int16_t *in0 = in_tm + tile_idx[0] * in_ch * 16;
        const int16_t *in1 = in_tm + tile_idx[1] * in_ch * 16;

        const int16_t *kernel0 = kernel_tm + outch_idx[0] * in_ch * 16;
        const int16_t *kernel1 = kernel_tm + outch_idx[1] * in_ch * 16;

        int32_t *dot0 = dot + out_idx * 16;
        int32_t *dot1 = dot0 + 16;

        for (int32_t cin_idx = 0; cin_idx < in_ch; ++cin_idx) {
            vint16m2_t vin0 = __riscv_vle16_v_i16m2(in0 + cin_idx * 16, 16);
            vint16m2_t vin1 = __riscv_vle16_v_i16m2(in1 + cin_idx * 16, 16);

            vint16m2_t vkernel0 = __riscv_vle16_v_i16m2(kernel0 + cin_idx * 16, 16);
            vint16m2_t vkernel1 = __riscv_vle16_v_i16m2(kernel1 + cin_idx * 16, 16);

            sum0 = __riscv_vwmacc_vv_i32m4(sum0, vin0, vkernel0, 16);
            sum1 = __riscv_vwmacc_vv_i32m4(sum1, vin1, vkernel1, 16);
        }

        __riscv_vse32_v_i32m4(dot0, sum0, 16);
        __riscv_vse32_v_i32m4(dot1, sum1, 16);
    }

    if (out_idx < ouput_cnt) {
        outch_idx[0] = out_idx % out_ch;
        tile_idx[0] = out_idx / out_ch;

        vint32m4_t sum0 = __riscv_vmv_v_x_i32m4(0, 16);

        const int16_t *in0 = in_tm + tile_idx[0] * in_ch * 16;
        const int16_t *kernel0 = kernel_tm + outch_idx[0] * in_ch * 16;
        int32_t *dot0 = dot + out_idx * 16;

        for (int32_t cin_idx = 0; cin_idx < in_ch; ++cin_idx) {
            vint16m2_t vin0 = __riscv_vle16_v_i16m2(in0 + cin_idx * 16, 16);
            vint16m2_t vkernel0 = __riscv_vle16_v_i16m2(kernel0 + cin_idx * 16, 16);
            sum0 = __riscv_vwmacc_vv_i32m4(sum0, vin0, vkernel0, 16);
        }

        __riscv_vse32_v_i32m4(dot0, sum0, 16);
    }

    return 0;
}

// input is [1, tiles, C_OUT, 16]
// output is [1, tiles×C_OUT, 8, 1]
// multiply with left hand matrix
//    [1,  1,  1,  0]
//    [0,  1, -1, -1]
static void trans_output_row_op(int32_t tiles, int32_t out_ch, const int32_t *dot, int32_t *output)
{
    uint32_t outch_idx[4];
    uint32_t tile_idx[4];

    uint32_t out_idx = 0;
    const int32_t ouput_cnt = tiles * out_ch;
    for (; out_idx + 3 < ouput_cnt; out_idx += 4) {
        {
            vuint32m1_t vidx = __riscv_vadd_vx_u32m1(__riscv_vid_v_u32m1(4), out_idx, 4);
            vuint32m1_t outch_vidx = __riscv_vremu_vx_u32m1(vidx, out_ch, 4);
            __riscv_vse32_v_u32m1(outch_idx, outch_vidx, 4);
            vuint32m1_t tile_vidx = __riscv_vdivu_vx_u32m1(vidx, out_ch, 4);
            __riscv_vse32_v_u32m1(tile_idx, tile_vidx, 4);
        }

        const int32_t *mat0 = dot + out_idx * 16;
        const int32_t *mat1 = mat0 + 16;
        const int32_t *mat2 = mat1 + 16;
        const int32_t *mat3 = mat2 + 16;

        int32_t *out0 = output + out_idx * 8;
        int32_t *out1 = out0 + 8;
        int32_t *out2 = out1 + 8;
        int32_t *out3 = out2 + 8;

        vint32m1_t row00 = __riscv_vle32_v_i32m1(mat0, 4);
        vint32m1_t row10 = __riscv_vle32_v_i32m1(mat1, 4);
        vint32m1_t row20 = __riscv_vle32_v_i32m1(mat2, 4);
        vint32m1_t row30 = __riscv_vle32_v_i32m1(mat3, 4);
        vint32m1_t row01 = __riscv_vle32_v_i32m1(mat0 + 4, 4);
        vint32m1_t row11 = __riscv_vle32_v_i32m1(mat1 + 4, 4);
        vint32m1_t row21 = __riscv_vle32_v_i32m1(mat2 + 4, 4);
        vint32m1_t row31 = __riscv_vle32_v_i32m1(mat3 + 4, 4);

        row00 = __riscv_vadd_vv_i32m1(row00, __riscv_vle32_v_i32m1(mat0 + 4, 4), 4);
        row10 = __riscv_vadd_vv_i32m1(row10, __riscv_vle32_v_i32m1(mat1 + 4, 4), 4);
        row20 = __riscv_vadd_vv_i32m1(row20, __riscv_vle32_v_i32m1(mat2 + 4, 4), 4);
        row30 = __riscv_vadd_vv_i32m1(row30, __riscv_vle32_v_i32m1(mat3 + 4, 4), 4);
        row01 = __riscv_vsub_vv_i32m1(row01, __riscv_vle32_v_i32m1(mat0 + 8, 4), 4);
        row11 = __riscv_vsub_vv_i32m1(row11, __riscv_vle32_v_i32m1(mat1 + 8, 4), 4);
        row21 = __riscv_vsub_vv_i32m1(row21, __riscv_vle32_v_i32m1(mat2 + 8, 4), 4);
        row31 = __riscv_vsub_vv_i32m1(row31, __riscv_vle32_v_i32m1(mat3 + 8, 4), 4);

        row00 = __riscv_vadd_vv_i32m1(row00, __riscv_vle32_v_i32m1(mat0 + 8, 4), 4);
        row10 = __riscv_vadd_vv_i32m1(row10, __riscv_vle32_v_i32m1(mat1 + 8, 4), 4);
        row20 = __riscv_vadd_vv_i32m1(row20, __riscv_vle32_v_i32m1(mat2 + 8, 4), 4);
        row30 = __riscv_vadd_vv_i32m1(row30, __riscv_vle32_v_i32m1(mat3 + 8, 4), 4);
        row01 = __riscv_vsub_vv_i32m1(row01, __riscv_vle32_v_i32m1(mat0 + 12, 4), 4);
        row11 = __riscv_vsub_vv_i32m1(row11, __riscv_vle32_v_i32m1(mat1 + 12, 4), 4);
        row21 = __riscv_vsub_vv_i32m1(row21, __riscv_vle32_v_i32m1(mat2 + 12, 4), 4);
        row31 = __riscv_vsub_vv_i32m1(row31, __riscv_vle32_v_i32m1(mat3 + 12, 4), 4);

        __riscv_vse32_v_i32m1(out0, row00, 4);
        __riscv_vse32_v_i32m1(out1, row10, 4);
        __riscv_vse32_v_i32m1(out2, row20, 4);
        __riscv_vse32_v_i32m1(out3, row30, 4);
        __riscv_vse32_v_i32m1(out0 + 4, row01, 4);
        __riscv_vse32_v_i32m1(out1 + 4, row11, 4);
        __riscv_vse32_v_i32m1(out2 + 4, row21, 4);
        __riscv_vse32_v_i32m1(out3 + 4, row31, 4);
    }

    for (; out_idx + 1 < ouput_cnt; out_idx += 2) {
        {
            vuint32m1_t vidx = __riscv_vadd_vx_u32m1(__riscv_vid_v_u32m1(2), out_idx, 2);
            vuint32m1_t outch_vidx = __riscv_vremu_vx_u32m1(vidx, out_ch, 2);
            __riscv_vse32_v_u32m1(outch_idx, outch_vidx, 2);
            vuint32m1_t tile_vidx = __riscv_vdivu_vx_u32m1(vidx, out_ch, 2);
            __riscv_vse32_v_u32m1(tile_idx, tile_vidx, 4);
        }

        const int32_t *mat0 = dot + out_idx * 16;
        const int32_t *mat1 = mat0 + 16;

        int32_t *out0 = output + out_idx * 8;
        int32_t *out1 = out0 + 8;

        vint32m1_t row00 = __riscv_vle32_v_i32m1(mat0, 4);
        vint32m1_t row10 = __riscv_vle32_v_i32m1(mat1, 4);
        vint32m1_t row01 = __riscv_vle32_v_i32m1(mat0 + 4, 4);
        vint32m1_t row11 = __riscv_vle32_v_i32m1(mat1 + 4, 4);

        row00 = __riscv_vadd_vv_i32m1(row00, __riscv_vle32_v_i32m1(mat0 + 4, 4), 4);
        row10 = __riscv_vadd_vv_i32m1(row10, __riscv_vle32_v_i32m1(mat1 + 4, 4), 4);
        row01 = __riscv_vsub_vv_i32m1(row01, __riscv_vle32_v_i32m1(mat0 + 8, 4), 4);
        row11 = __riscv_vsub_vv_i32m1(row11, __riscv_vle32_v_i32m1(mat1 + 8, 4), 4);

        row00 = __riscv_vadd_vv_i32m1(row00, __riscv_vle32_v_i32m1(mat0 + 8, 4), 4);
        row10 = __riscv_vadd_vv_i32m1(row10, __riscv_vle32_v_i32m1(mat1 + 8, 4), 4);
        row01 = __riscv_vsub_vv_i32m1(row01, __riscv_vle32_v_i32m1(mat0 + 12, 4), 4);
        row11 = __riscv_vsub_vv_i32m1(row11, __riscv_vle32_v_i32m1(mat1 + 12, 4), 4);

        __riscv_vse32_v_i32m1(out0, row00, 4);
        __riscv_vse32_v_i32m1(out1, row10, 4);
        __riscv_vse32_v_i32m1(out0 + 4, row01, 4);
        __riscv_vse32_v_i32m1(out1 + 4, row11, 4);
    }

    if (out_idx < ouput_cnt) {
        outch_idx[0] = out_idx % out_ch;
        tile_idx[0] = out_idx / out_ch;

        const int32_t *mat0 = dot + out_idx * 16;
        int32_t *out0 = output + out_idx * 8;

        vint32m1_t row00 = __riscv_vle32_v_i32m1(mat0, 4);
        vint32m1_t row01 = __riscv_vle32_v_i32m1(mat0 + 4, 4);

        row00 = __riscv_vadd_vv_i32m1(row00, __riscv_vle32_v_i32m1(mat0 + 4, 4), 4);
        row01 = __riscv_vsub_vv_i32m1(row01, __riscv_vle32_v_i32m1(mat0 + 8, 4), 4);

        row00 = __riscv_vadd_vv_i32m1(row00, __riscv_vle32_v_i32m1(mat0 + 8, 4), 4);
        row01 = __riscv_vsub_vv_i32m1(row01, __riscv_vle32_v_i32m1(mat0 + 12, 4), 4);

        __riscv_vse32_v_i32m1(out0, row00, 4);
        __riscv_vse32_v_i32m1(out0 + 4, row01, 4);
    }
}

// input is [1, tiles×C_OUT, 8, 1]
// output is [1, tiles×C_OUT, 4, 1]
static void trans_output_col_op(int32_t tiles, int32_t out_ch, const int32_t *buffer, int32_t *output)
{
    size_t avl = tiles * out_ch * 2, vl;
    const ptrdiff_t bstride = 4 * sizeof(int32_t);

    const int32_t *col0 = buffer;
    const int32_t *col1 = col0 + 1;
    const int32_t *col2 = col1 + 1;
    const int32_t *col3 = col2 + 1;

    for (; (vl = __riscv_vsetvl_e32m4(avl)) > 0; avl -= vl) {
        vint32m4_t vcol0 = __riscv_vlse32_v_i32m4(col0, bstride, vl);
        vint32m4_t vcol1 = __riscv_vlse32_v_i32m4(col1, bstride, vl);
        vint32m4_t vcol2 = __riscv_vlse32_v_i32m4(col2, bstride, vl);
        vint32m4_t vcol3 = __riscv_vlse32_v_i32m4(col3, bstride, vl);

        col0 += 4 * vl;
        col1 += 4 * vl;
        col2 += 4 * vl;
        col3 += 4 * vl;

        vint32m4x2_t v_tuple;
        v_tuple = __riscv_vset_v_i32m4_i32m4x2(v_tuple, 0, __riscv_vadd_vv_i32m4(__riscv_vadd_vv_i32m4(vcol0, vcol1, vl), vcol2, vl));
        v_tuple = __riscv_vset_v_i32m4_i32m4x2(v_tuple, 1, __riscv_vsub_vv_i32m4(__riscv_vsub_vv_i32m4(vcol1, vcol2, vl), vcol3, vl));
        __riscv_vsseg2e32_v_i32m4x2(output, v_tuple, vl);
        output += vl * 2;
    }
}

static int32_t convolve_3x3_s8_wg23_trans_output(const Tile *output_shape, const int32_t out_offset, const int32_t out_activation_min,
                                                 const int32_t out_activation_max, const int *dot_dims, const int32_t *dot,
                                                 const int32_t *output_mult_ptr, const int32_t *output_shift_ptr, const int *bias_dims,
                                                 const int32_t *bias_data, const int *output_dims, int32_t *buffer, int8_t *output_data)
{
    // output_dims->n is not used and assumed to be 1
    // shape of dot is [N, H, W, C] = [1, tiles, C_OUT, 16]
    const int32_t tile_w = output_shape->w >> 1;
    const int32_t tile_h = output_shape->h >> 1;
    const int32_t tiles = tile_w * tile_h;
    const int32_t out_ch = dot_dims[1];
    const int32_t *bias_data_ptr = bias_data;

    const ptrdiff_t bstride = 4 * sizeof(int32_t);

    // A = [1,  1,  1,  0]
    //     [0,  1, -1, -1]

    // A @ dot
    trans_output_row_op(tiles, out_ch, dot, buffer);

    // A @ dot @ A^T
    trans_output_col_op(tiles, out_ch, buffer, buffer);

    for (int32_t h_idx = 0; h_idx < tile_h; ++h_idx) {
        for (int32_t w_idx = 0; w_idx < tile_w; ++w_idx) {
            const int32_t tile_idx = h_idx * tile_w + w_idx;
            const int32_t *tile_base = buffer + tile_idx * out_ch * 4;

            _Bool w_valid = w_idx * 2 + 1 < output_dims[1];
            _Bool h_valid = h_idx * 2 + 1 < output_dims[2];

            int8_t *out00 = output_data + ((h_idx * 2 + 0) * output_dims[1] + w_idx * 2 + 0) * out_ch;
            int8_t *out01 = output_data + ((h_idx * 2 + 0) * output_dims[1] + w_idx * 2 + 1) * out_ch;
            int8_t *out10 = output_data + ((h_idx * 2 + 1) * output_dims[1] + w_idx * 2 + 0) * out_ch;
            int8_t *out11 = output_data + ((h_idx * 2 + 1) * output_dims[1] + w_idx * 2 + 1) * out_ch;

            size_t avl = out_ch, vl;
            const int32_t *bias_ptr = bias_data_ptr;
            for (; (vl = __riscv_vsetvl_e32m4(avl)) > 0; avl -= vl) {
                vint32m4_t d00 = __riscv_vlse32_v_i32m4(tile_base, bstride, vl);
                vint32m4_t d01 = __riscv_vlse32_v_i32m4(tile_base + 1, bstride, vl);
                vint32m4_t d10 = __riscv_vlse32_v_i32m4(tile_base + 2, bstride, vl);
                vint32m4_t d11 = __riscv_vlse32_v_i32m4(tile_base + 3, bstride, vl);
                tile_base += 4 * vl;

                // add bias
                {
                    vint32m4_t v_bias = __riscv_vle32_v_i32m4(bias_ptr, vl);
                    bias_ptr += vl;
                    d00 = __riscv_vadd_vv_i32m4(d00, v_bias, vl);
                    d01 = __riscv_vadd_vv_i32m4(d01, v_bias, vl);
                    d10 = __riscv_vadd_vv_i32m4(d10, v_bias, vl);
                    d11 = __riscv_vadd_vv_i32m4(d11, v_bias, vl);
                }

                vint32m4_t shift = __riscv_vsub_vx_i32m4(__riscv_vle32_v_i32m4(output_shift_ptr, vl), 2, vl);
                {
                    // left shift
                    vuint32m4_t left_shift = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax_vx_i32m4(shift, 0, vl));
                    d00 = __riscv_vsll_vv_i32m4(d00, left_shift, vl);
                    d01 = __riscv_vsll_vv_i32m4(d01, left_shift, vl);
                    d10 = __riscv_vsll_vv_i32m4(d10, left_shift, vl);
                    d11 = __riscv_vsll_vv_i32m4(d11, left_shift, vl);
                }

                // requantize multiply
                {
                    vint32m4_t mult = __riscv_vle32_v_i32m4(output_mult_ptr, vl);
                    d00 = __riscv_vsmul_vv_i32m4(d00, mult, __RISCV_VXRM_RNU, vl);
                    d01 = __riscv_vsmul_vv_i32m4(d01, mult, __RISCV_VXRM_RNU, vl);
                    d10 = __riscv_vsmul_vv_i32m4(d10, mult, __RISCV_VXRM_RNU, vl);
                    d11 = __riscv_vsmul_vv_i32m4(d11, mult, __RISCV_VXRM_RNU, vl);
                }

                {
                    // right shift
                    vuint32m4_t right_shift = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vneg_v_i32m4(__riscv_vmin_vx_i32m4(shift, 0, vl), vl));
                    d00 = __riscv_vssra_vv_i32m4(d00, right_shift, __RISCV_VXRM_RNU, vl);
                    d01 = __riscv_vssra_vv_i32m4(d01, right_shift, __RISCV_VXRM_RNU, vl);
                    d10 = __riscv_vssra_vv_i32m4(d10, right_shift, __RISCV_VXRM_RNU, vl);
                    d11 = __riscv_vssra_vv_i32m4(d11, right_shift, __RISCV_VXRM_RNU, vl);
                }

                d00 = __riscv_vadd_vx_i32m4(d00, out_offset, vl);
                d01 = __riscv_vadd_vx_i32m4(d01, out_offset, vl);
                d10 = __riscv_vadd_vx_i32m4(d10, out_offset, vl);
                d11 = __riscv_vadd_vx_i32m4(d11, out_offset, vl);

                // clip result
                vint8m1_t d00_i8 = __riscv_vnclip_wx_i8m1(__riscv_vnclip_wx_i16m2(d00, 0, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl);
                vint8m1_t d01_i8 = __riscv_vnclip_wx_i8m1(__riscv_vnclip_wx_i16m2(d01, 0, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl);
                vint8m1_t d10_i8 = __riscv_vnclip_wx_i8m1(__riscv_vnclip_wx_i16m2(d10, 0, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl);
                vint8m1_t d11_i8 = __riscv_vnclip_wx_i8m1(__riscv_vnclip_wx_i16m2(d11, 0, __RISCV_VXRM_RNU, vl), 0, __RISCV_VXRM_RNU, vl);

                // store result
                __riscv_vse8_v_i8m1(out00, d00_i8, vl);
                out00 += vl;
                if (w_valid) {
                    __riscv_vse8_v_i8m1(out01, d01_i8, vl);
                    out01 += vl;
                }
                if (h_valid) {
                    __riscv_vse8_v_i8m1(out10, d10_i8, vl);
                    out10 += vl;
                }
                if (w_valid && h_valid) {
                    __riscv_vse8_v_i8m1(out11, d11_i8, vl);
                    out11 += vl;
                }
            }
        }
    }
    return 0;
}

int ConvInteger_rvv(struct onnx_node_t *n)
{
    const struct operator_pdata_t *pdat = (struct operator_pdata_t *)n->priv;
    if (pdat->stride.w != 1 || pdat->stride.h != 1 || pdat->ctx.buf == NULL || pdat->dilation.h != 1 || pdat->dilation.w != 1) {
        return -1;
    }
    const struct onnx_tensor_t *filter = n->inputs[1]; // shape [output_ch, kernel_h, kernel_w, input_ch]
    if (filter->dims[1] != 3 || filter->dims[2] != 3) {
        return -1;
    }

    const struct onnx_tensor_t *input = n->inputs[0];    // shape [batch, input_h, input_w, input_ch]
    const struct onnx_tensor_t *bias = n->inputs[2];     // shape [output_ch]
    const struct onnx_tensor_t *multiply = n->inputs[3]; // shape [output_ch]
    const struct onnx_tensor_t *shift = n->inputs[4];    // shape [output_ch]

    struct onnx_tensor_t *output = n->outputs[0]; // shape [batch, output_h, output_w, output_ch]

    const int32_t batch = input->dims[3];
    const int32_t out_inner_w = (output->dims[1] + 1) / 2 * 2;
    const int32_t out_inner_h = (output->dims[2] + 1) / 2 * 2;
    const int32_t tile_w = out_inner_w >> 1;
    const int32_t tile_h = out_inner_h >> 1;
    const int32_t tiles = tile_w * tile_h;
    const int32_t in_ch = input->dims[0];
    const int32_t out_ch = output->dims[0];
    const int8_t *input_data = (int8_t *)input->datas;
    const int *input_dims = input->dims;
    int8_t *output_data = (int8_t *)output->datas;
    const int *output_dims = output->dims;
    Tile output_shape = {out_inner_w, out_inner_h};

    int32_t in_preprocess_dims[4] = {in_ch, out_inner_w + 2, out_inner_h + 2, 1};
    // split buffer
    const int32_t in_tm_sz = tiles * 16 * in_ch * sizeof(int16_t) + in_ch * 16 * sizeof(int16_t);
    const int32_t kernel_tm_sz = 16 * out_ch * in_ch * sizeof(int16_t);
    const int32_t dot_sz = tiles * 16 * out_ch * sizeof(int32_t);
    int16_t *in_tm = (int16_t *)pdat->ctx.buf;
    int16_t *kernel_tm = (int16_t *)((char *)pdat->ctx.buf + in_tm_sz);
    int32_t *dot = (int32_t *)((char *)pdat->ctx.buf + in_tm_sz + kernel_tm_sz);

    // transform kernel shape of kernel_tm is [N, H, W, C] = [1, C_OUT, C_IN, 16]
    // nmsis_nn_dims kernel_tm_dims;
    // kernel_tm_dims.n = 1;
    // kernel_tm_dims.h = out_ch;
    // kernel_tm_dims.w = in_ch;
    // kernel_tm_dims.c = 16;
    const int32_t kernel_tm_dims[4] = {16, in_ch, out_ch, 1};
    int status = convolve_3x3_s8_wg23_trans_kernel(filter->dims, filter->datas, kernel_tm);
    if (status != 0) {
        return status;
    }

    for (int32_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
        const int8_t *in_preprocess;
        const int8_t *batch_in = input_data + input_dims[2] * input_dims[1] * in_ch * batch_idx;
        int8_t *buffer = (int8_t *)((char *)pdat->ctx.buf + in_tm_sz + kernel_tm_sz + dot_sz);
        if (pdat->padding.h != 0 || pdat->padding.w != 0 || output_dims[2] % 2 || output_dims[1] % 2 || input_dims[2] < output_dims[2] + 3 - 1 ||
            input_dims[1] < output_dims[1] + 3 - 1) {
            // need padding
            status =
                convolve_3x3_s8_wg23_pad_input(input_dims, pdat->padding.w, pdat->padding.h, pdat->input_offset, batch_in, &output_shape, buffer);
            if (status != 0) {
                return status;
            }
            in_preprocess = buffer;
        } else {
            in_preprocess = batch_in;
        }

        // transform input
        // nmsis_nn_dims in_tm_dims;
        // in_tm_dims.n = 1;
        // in_tm_dims.h = tiles;
        // in_tm_dims.w = in_ch;
        // in_tm_dims.c = 16;
        const int32_t in_tm_dims[4] = {16, in_ch, tiles, 1};
        status = convolve_3x3_s8_wg23_trans_input(in_preprocess_dims, in_preprocess, in_tm);
        if (status != 0) {
            return status;
        }

        // dot product
        // nmsis_nn_dims dot_dims;
        // dot_dims.n = 1;
        // dot_dims.h = tiles;
        // dot_dims.w = out_ch;
        // dot_dims.c = 16;
        const int dot_dims[4] = {16, out_ch, tiles, 1};
        status = riscv_convolve_3x3_s8_wg23_dot(in_tm_dims, in_tm, kernel_tm_dims, kernel_tm, dot_dims, dot);
        if (status != 0) {
            return status;
        }

        // transform output and crop padding
        int8_t *perbatch_out = output_data + output_dims[2] * output_dims[1] * output_dims[0] * batch_idx;
        status =
            convolve_3x3_s8_wg23_trans_output(&output_shape, pdat->output_offset, pdat->activation.min, pdat->activation.max, dot_dims, dot,
                                              multiply->datas, shift->datas, bias->dims, bias->datas, output_dims, (int32_t *)buffer, perbatch_out);
        if (status != 0) {
            return status;
        }
    }

    /* Return to application */
    return 0;
}

void *GenerateConvIntegerParam(int32_t in_offset, int32_t out_offset, int32_t stride_w, int32_t stride_h, int32_t dilation_w, int32_t dilation_h,
                               int32_t pad_w, int32_t pad_h, int32_t activation_min, int32_t activation_max, const struct onnx_tensor_t *input,
                               const struct onnx_tensor_t *filter, const struct onnx_tensor_t *output, _Bool rvv)
{
    struct operator_pdata_t *pdat = (struct operator_pdata_t *)MALLOC_ASSERT(sizeof(struct operator_pdata_t));
    pdat->input_offset = in_offset;
    pdat->output_offset = out_offset;
    pdat->stride.w = stride_w;
    pdat->stride.h = stride_h;
    pdat->dilation.w = dilation_w;
    pdat->dilation.h = dilation_h;
    pdat->padding.w = pad_w;
    pdat->padding.h = pad_h;
    pdat->activation.min = activation_min;
    pdat->activation.max = activation_max;

    if (rvv && filter->dims[1] == 3 && filter->dims[2] == 3) {
        // allocate buffer for rvv
        int32_t in_pad = 0;
        const int32_t out_inner_w = (output->dims[1] + 1) / 2 * 2;
        const int32_t out_inner_h = (output->dims[2] + 1) / 2 * 2;
        const int32_t tile_w = out_inner_w >> 1;
        const int32_t tile_h = out_inner_h >> 1;
        const int32_t tiles = tile_w * tile_h;
        const int32_t in_ch = input->dims[0];
        const int32_t out_ch = filter->dims[3];

        // whether copy to padding
        // 1. when padding is not zero
        // 2. when output dim is not multiple of 2
        // 3. when input dim is not match to output dim
        if (pad_h != 0 || pad_w != 0 || output->dims[2] % 2 || output->dims[1] % 2 || input->dims[2] < output->dims[2] + 3 - 1 ||
            input->dims[1] < output->dims[1] + 3 - 1) {
            // need padding
            int32_t in_sz = (out_inner_w + 2) * (out_inner_h + 2) * in_ch * sizeof(int8_t);
            int32_t out_sz = tiles * out_ch * sizeof(int32_t) * 8;
            // the same memory is also used to store output transform result
            in_pad = in_sz > out_sz ? in_sz : out_sz;
        }

        /* there are three buffers needed for wg23
         * 1. to store the transform result of input data, in_tm; in_tm need extra memory for transform data layout.
         * 2. to store the transform result of kernel data, kernel_tm
         * 3. to store the dot product of in_tm and kernel_tm.(_tm means transform)
         *
         * 16 elements, because a input tile is 4x4
         */
        const int32_t in_tm_sz = tiles * 16 * in_ch * sizeof(int16_t) + in_ch * 16 * sizeof(int16_t);
        const int32_t kernel_tm_sz = 16 * out_ch * in_ch * sizeof(int16_t);
        const int32_t dot_sz = tiles * 16 * out_ch * sizeof(int32_t);
        pdat->ctx.buf_size = in_tm_sz + kernel_tm_sz + dot_sz + in_pad;
        pdat->ctx.buf = MALLOC_ASSERT(pdat->ctx.buf_size);
    } else {
        // allocate buffer for non-rvv
        const int32_t rhs_cols = filter->dims[1] * filter->dims[2] * filter->dims[0];
        const int32_t remainder = rhs_cols % 4;
        const int32_t aligned_rhs_cols = remainder != 0 ? rhs_cols + 4 - remainder : rhs_cols;
        pdat->ctx.buf_size = (2 * aligned_rhs_cols) * (int32_t)sizeof(int16_t);
        pdat->ctx.buf = MALLOC_ASSERT(pdat->ctx.buf_size);
    }
    return pdat;
}

void FreeConvIntegerParam(void **pdat)
{
    struct operator_pdata_t *_pdat = (struct operator_pdata_t *)*pdat;
    if (_pdat->ctx.buf != NULL) {
        free(_pdat->ctx.buf);
        _pdat->ctx.buf = NULL;
        _pdat->ctx.buf_size = 0;
    }
    free(*pdat);
    *pdat = NULL;
}