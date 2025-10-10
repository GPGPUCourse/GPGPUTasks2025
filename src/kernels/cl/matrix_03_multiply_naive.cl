__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void
matrix_03_multiply_naive(
    __global const float* a, // rows=h x cols=k
    __global const float* b, // rows=k x cols=w
    __global float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    const unsigned int column = get_global_id(0);
    const unsigned int row = get_global_id(1);

    float result = 0;
    for (unsigned int i = 0; i < k; ++i) {
        result += a[row * k + i] * b[i * w + column];
    }

    c[row * w + column] = result;
}
