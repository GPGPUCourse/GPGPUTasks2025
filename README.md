В этом репозитории предложены задания для курса по вычислениям на видеокартах 2025.

[Остальные задания](https://github.com/GPGPUCourse/GPGPUTasks2025/).

# Задание 1. A + B сложение матриц (*coalesced memory access*)

[![Build Status](https://github.com/GPGPUCourse/GPGPUTasks2025/actions/workflows/cmake.yml/badge.svg?branch=task01&event=push)](https://github.com/GPGPUCourse/GPGPUTasks2025/actions/workflows/cmake.yml)

В этом задании вам предлагается выбрать API (OpenCL, CUDA или Vulkan) и попробовать написать два кернела поэлементного сложения двумерных матриц:

1) один кернел максимально эффективный с точки зрения *coalesced memory access*
2) и второй кернел - максимально неэффективный

Таким образом хочется провести эксперимент и на практике проверить - а есть ли разница? И если есть - то насколько эта разница в достигнутой пропускной способности памяти заметна?

Для примера в [исходном коде](/src/main_aplusb.cpp) уже реализованы кернелы сложения двух массивов на всех трех API ([OpenCL](src/kernels/cl/aplusb.cl), [CUDA](src/kernels/cu/aplusb.cu), [Vulkan](src/kernels/vk/aplusb.comp)).

Установка зависимостей
========================================

Вам надо установить из соответствующей папки ```scripts/<моя OS>```:

1) ```install_dependencies``` - Некоторые зависимости: googletest, для Windows и macOS - дополнительно clang.
2) ```install_vulkan_sdk``` - Vulkan SDK. Даже если вы не будете писать на Vulkan, и если пропустите вероятные будущие домашние задания на Vulkan - проект не соберется без Vulkan SDK.
3) *опционально* ```install_opencl_cpu_driver``` - OpenCL CPU драйвер на Linux (Intel драйвер, работает и на AMD CPU)
4) *опционально* ```install_vulkan_cpu_driver``` - Vulkan CPU драйвер на Linux (проект mesa)
5) *опционально* ```install_cuda_sdk``` - CUDA SDK, в случае если у вас NVIDIA и хотите писать на CUDA (рекомендуется), не забудьте после этого добавить в CMake options: ```-DGPU_CUDA_SUPPORT=ON```

Если у вас возникли проблемы - обратитесь за помощью к департаменту технической поддержки в чате курса или заведите **Issue**.

Проверка окружения и начало выполнения задания
==============================================

Это инструкция с чистого листа, но если хотите и понимаете как, то вы можете продолжать работу в уже скачанном проекте - вам нужно будет переключится на ветку ```task01``` и внимательно посмотреть на шаг про CMake options.

1. Сделайте fork этого репозитория
2. ``git clone ВАШ_ФОРК_РЕПОЗИТОРИЯ``
3. ``cd GPGPUTasks2025``
4. ``git checkout task01``
5. ``mkdir build``
6. ``cd build``
7. Здесь возможны варианты:
 
7.1. **Linux** - ```cmake -DGPU_CUDA_SUPPORT=ON .. ``` (или без ```-DGPU_CUDA_SUPPORT=ON```)

7.2. **Windows** - укажите в CLion->File->Settings->Build->CMake->CMake options: ```-DCMAKE_TOOLCHAIN_FILE=C:\Users\<USERNAME>\.vcpkg-clion\vcpkg\scripts\buildsystems\vcpkg.cmake -DSPIR_CLANG_BIN="C:\Program Files\LLVM\bin\clang.exe" -DGPU_CUDA_SUPPORT=ON``` (или без ```-DGPU_CUDA_SUPPORT=ON```)

7.3. **macOS** - вероятно что-то очень похожее на Linux, напишите если будут трудности

8. ``make -j8``
9. ``./libs/gpu/libgpu_test`` прогонит unit-test-ы покрывающие OpenCL и Vulkan
10. ``./main_aplusb`` убедится что программа-пример работает корректно на OpenCL
11. ``./main_aplusb_matrix`` убедится что программа-задание запускается и пока что не выполнена

Задание
=======

0. Сделать fork проекта если еще не сделали
1. Изучить ```src/main_aplusb.cpp``` и ```src/kernels/cl/aplusb.cl``` (если хочется использовать не OpenCL - изменить используемое API в main_aplusb.cpp - см. там **TODO**)
2. Прочитать все комментарии подряд и выполнить все **TODO** в файле ``src/main_aplusb_matrix.cpp`` и реализовать два кернела (API на ваш выбор). Для разработки под Linux рекомендуется использовать CLion. Под Windows рекомендуется использовать CLion+MSVC. Также под Windows можно использовать Visual Studio Community.
3. Отправить **Pull-request** с названием ```Task01 <Имя> <Фамилия> <Аффиляция>```. **Аффиляция** - ваш ВУЗ (например SPbU/ITMO или HSE) или ваше место работы. Мне интересно узнать кто откуда.
4. В тексте **PR** укажите вывод программы при исполнении на сервере Github CI (Github Actions) и на вашем компьютере (в **pre**-тэгах, чтобы сохранить форматирование, см. [пример](https://raw.githubusercontent.com/GPGPUCourse/GPGPUTasks2025/task01/.github/pull_request_example.md)). И ваш бранч должен называться так же, как и у меня - **task01**.
5. Убедиться что Github CI (Github Actions) смог скомпилировать ваш код и что все хорошо, при отправке первого задания CI может не запуститься пока я вручную не нажму ```Approve``` - если я этого не сделал в течение суток - напомните мне пожалуйста в чате курса
6. Ждать комментарии проверки

**Дедлайн**: 23:59 29 сентября. Но очень советую сделать в ближайшую неделю. Задание простое, если вы его будете откладывать - скорее-всего дальше будет снежный ком, т.к. в какой-то момент начнутся сложные задания. В целом в рамках курса будет много домашних заданий и будет тяжело.

Как работать под Windows
========================

1. Используйте **64-битный компилятор**, т.е. [amd64](/.figures/clion_msvc_settings.png), а не x86. (Если при запуске видите ``Invalid Parameter - 100``, то вы все еще используете 32-битный компилятор)
2. Рекомендуется использовать CLion+MSVC.
3. Можно использовать Visual Studio 2017 Community или новее, она поддерживает CMake-проекты (``File`` -> ``Open`` -> ``Cmake...``). Разве что передавать аргументы запускаемой программе [неудобно](https://docs.microsoft.com/en-us/cpp/ide/cmake-tools-for-visual-cpp?view=vs-2017#configure-cmake-debugging-sessions). Но может быть уже сделали GUI.
