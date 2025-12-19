#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void denoise(
    int width, int height,
    __global const int* face_id,
    __global const float* color,
    __global float* out,
    __global const float* var_in,
    __global float* var_out,
    int iter)
{
    // что-то похожее на упрощённый вариант "Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination"
    // упрощения вызваны как сложностью, так и нежеланием возвращать нормали и глубины из трассировщика лучей чтобы не портить бенчмарки
    // (интересно было бы попробовать вместо face_id коды мортона от векторов нормали, но это надо что-то делать)
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x >= width || y >= height)
        return;
    int ci = y * width + x;
    int cur_face = face_id[ci];
    float cur_color = color[ci];
    if(iter == -1)
    {
        float s1 = 0, s2 = 0;
        for(int dx = -3; dx <= 3; dx++)
        for(int dy = -3; dy <= 3; dy++) {
            int ni = clamp(y + dy, 0, height) * width + clamp(x + dx, 0, width);
            s1 += color[ni];  
            s2 += color[ni] * color[ni];
        }
        float var = s2 / 49.0f - (s1 / 49.0f) * (s1 / 49.0f);
        var_out[ci] = max(var, 0.0f);
        return;
    }
    float var = 0;
    for(int dx = -1; dx <= 1; dx++)
    for(int dy = -1; dy <= 1; dy++) {
        var += var_in[clamp(y + dy, 0, height) * width + clamp(x + dx, 0, width)] / ((dx == 0 ? 1.0f : 2.0f) * (dy == 0 ? 1.0f : 2.0f) * 4.0f);
    }
    float std = sqrt(var);
    float s1 = 0, s2 = 0, v1 = 0;
    for(int dx = -2; dx <= 2; dx++)
    for(int dy = -2; dy <= 2; dy++) {
        const float ks[5] = { 1.0f/16, 1.0f/4, 3.0f/8, 1.0f/4, 1.0f/16 };
        int ni = clamp(y + (dy << iter), 0, height) * width + clamp(x + (dx << iter), 0, width);
        float ex = exp(-fabs(cur_color - color[ni]) / (16 * std + 0.00001));
        float kw = face_id[ni] == cur_face ? 1.0f : 0.15f;
        kw *= (ks[dx + 2] * ks[dy + 2]);
        kw *= ex;
        s1 += color[ni] * kw;
        s2 += kw;
        v1 += var_in[ni] * kw * kw;
    }
    out[ci] = s1 / s2;
    var_out[ci] = v1 / (s2 * s2);
}
