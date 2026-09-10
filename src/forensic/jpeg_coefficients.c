/* Minimal libjpeg coefficient reader. Keep fatal libjpeg errors inside C. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <limits.h>
#include <jpeglib.h>

typedef struct {
    unsigned char *bins;
    short *coefficients;
    int height, width, rows, cols;
    unsigned short qtable[64];
    char error[256];
} jpeg_result;

struct reader_error { struct jpeg_error_mgr base; jmp_buf jump; };
static void reader_fail(j_common_ptr info) {
    struct reader_error *err = (struct reader_error *)info->err;
    longjmp(err->jump, 1);
}

int read_jpeg_bins(const unsigned char *bytes, unsigned long length, int include_coefficients, jpeg_result *out) {
    struct jpeg_decompress_struct *info = calloc(1, sizeof(*info));
    struct reader_error err;
    jvirt_barray_ptr *arrays;
    jpeg_component_info *component;
    int by, bx, u, v, value;
    memset(out, 0, sizeof(*out));
    if (!info) { strcpy(out->error, "JPEG decoder allocation failed"); return 0; }
    info->err = jpeg_std_error(&err.base);
    err.base.error_exit = reader_fail;
    if (setjmp(err.jump)) {
        (*info->err->format_message)((j_common_ptr)info, out->error);
        jpeg_destroy_decompress(info);
        free(info); free(out->bins); out->bins = NULL; free(out->coefficients); out->coefficients = NULL;
        return 0;
    }
    jpeg_create_decompress(info);
    /* Native coefficients are small enough for RAM; avoid libjpeg disk backing. */
    info->mem->max_memory_to_use = LONG_MAX;
    jpeg_mem_src(info, bytes, length);
    jpeg_read_header(info, TRUE);
    if (info->jpeg_color_space != JCS_YCbCr && info->jpeg_color_space != JCS_GRAYSCALE) {
        strcpy(out->error, "JPEG must contain YCbCr or grayscale components");
        jpeg_destroy_decompress(info); free(info); return 0;
    }
    arrays = jpeg_read_coefficients(info);
    component = &info->comp_info[0];
    out->height = (int)info->image_height; out->width = (int)info->image_width;
    out->rows = (int)component->height_in_blocks * 8;
    out->cols = (int)component->width_in_blocks * 8;
    out->bins = malloc((size_t)out->rows * out->cols);
    if (include_coefficients) out->coefficients = malloc((size_t)out->rows * out->cols * sizeof(short));
    if (!out->bins || (include_coefficients && !out->coefficients)) {
        strcpy(out->error, "JPEG coefficient allocation failed");
        free(out->bins); free(out->coefficients); out->bins = NULL; out->coefficients = NULL;
        jpeg_destroy_decompress(info); free(info); return 0;
    }
    for (u = 0; u < 64; u++)
        out->qtable[u] = info->quant_tbl_ptrs[component->quant_tbl_no]->quantval[u];
    for (by = 0; by < (int)component->height_in_blocks; by++) {
        JBLOCKARRAY row = (*info->mem->access_virt_barray)((j_common_ptr)info, arrays[0], by, 1, FALSE);
        for (bx = 0; bx < (int)component->width_in_blocks; bx++)
            for (u = 0; u < 8; u++) for (v = 0; v < 8; v++) {
                value = row[0][bx][u * 8 + v];
                if (out->coefficients) out->coefficients[(size_t)(by * 8 + u) * out->cols + bx * 8 + v] = (short)value;
                if (value < 0) value = -value;
                out->bins[(size_t)(by * 8 + u) * out->cols + bx * 8 + v] = (unsigned char)(value > 20 ? 20 : value);
            }
    }
    jpeg_finish_decompress(info); jpeg_destroy_decompress(info); free(info);
    return 1;
}
void free_jpeg_bins(jpeg_result *out) { free(out->bins); out->bins = NULL; free(out->coefficients); out->coefficients = NULL; }
