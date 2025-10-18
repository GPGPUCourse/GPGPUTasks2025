#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void aplusb(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                            unsigned int  n)
{
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    if (index == 0) {
        // из кернела можно печатать текст в консоль, буфер для текста ограничен в размере,
        // кроме того в моделе массового параллелизма у нас обычно очень много workItems,
        // поэтому если каждый будет печатать сообщение - буфер быстро переполниться, а разобраться в сообщениях может быть тяжело
        // поэтому это сообщение выводится только для первого workItem (index == 0)
        printf("OpenCL printf test in aplusb.cl kernel! a[index]=%d b[index]=%d \n", a[index], b[index]);
    }

    // rassert-ы - это способ легко проверить инвариант, если вдруг он будет нарушен - в консоль будет напечатан код этого инварианта
    // Не забудьте включить rassert в defines.h файле через RASSERT_ENABLED 1 (он выключен по умолчанию, не закомитьте случайно)
    // Попробуйте заполнить буферы в CPU коде другими значениями и запустить кернел - обнаружит ли он что инварианты нарушены?
    rassert(a[index] == 3 * (index + 5) + 7, 43562543223);
    rassert(b[index] == 11 * (index + 13) + 17, 546365435);

    c[index] = a[index] + b[index];
}
