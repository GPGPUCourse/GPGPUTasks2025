#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void mandelbrot_int(
    __global char* results,
    const unsigned int width,
    const unsigned int height,
    const int fromX_fp,
    const int fromY_fp,
    const int sizeX_fp,
    const int sizeY_fp,
    const unsigned int iters,
    const unsigned int isSmoothing)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    if (i >= width || j >= height) return;

    const int FP_SHIFT = 16;
    const long FP_ONE = 1L << FP_SHIFT;
    const long HALF_PIXEL = 1L << (FP_SHIFT - 1);

    const long threshold_fp = (long)256 << FP_SHIFT;
    const long threshold2_fp = (threshold_fp * threshold_fp) >> FP_SHIFT;

    int frac_x = (int)(((long)i << FP_SHIFT) + HALF_PIXEL) / (int)width;
    int frac_y = (int)(((long)j << FP_SHIFT) + HALF_PIXEL) / (int)height;

    int x0_fp = fromX_fp + (int)((((long)frac_x * (long)sizeX_fp) >> FP_SHIFT));
    int y0_fp = fromY_fp + (int)((((long)frac_y * (long)sizeY_fp) >> FP_SHIFT));

    int x_fp = x0_fp;
    int y_fp = y0_fp;

    unsigned int iter = 0;
    long mag2_fp = 0;

    for (; iter < iters; ++iter) {
        long x2_fp = (((long)x_fp * (long)x_fp) >> FP_SHIFT);
        long y2_fp = (((long)y_fp * (long)y_fp) >> FP_SHIFT);
        long xy_fp = (((long)x_fp * (long)y_fp) >> FP_SHIFT);

        int x_new_fp = (int)(x2_fp - y2_fp + x0_fp);
        int y_new_fp = (int)((xy_fp << 1) + y0_fp);

        x_fp = x_new_fp;
        y_fp = y_new_fp;

        mag2_fp = x2_fp + y2_fp;

        if (mag2_fp > threshold2_fp) {
            break;
        }
    }

    long result_fp = ((long)iter << FP_SHIFT);

    if (isSmoothing && iter != iters) {
        long diff = threshold2_fp - mag2_fp;
        if (diff < 0) diff = 0;
        long frac_fp = 0;
        if (threshold2_fp != 0) {
            frac_fp = (diff << FP_SHIFT) / threshold2_fp;
        }
        result_fp += frac_fp;
    }

    long out_val = 0;
    if (iters != 0) {
        out_val = (result_fp * 127) / (((long)iters) << FP_SHIFT);
    }
    if (out_val < 0) out_val = 0;
    if (out_val > 127) out_val = 127;

    int idx = j * width + i;
    results[idx] = (char)(out_val & 0xFF);
}
