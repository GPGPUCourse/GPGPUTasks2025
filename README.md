В этом репозитории предложены задания для курса по вычислениям на видеокартах 2025.

[Остальные задания](https://github.com/GPGPUCourse/GPGPUTasks2025/).

# Задание 0. Вводное.

[![Build Status](https://github.com/GPGPUCourse/GPGPUTasks2025/actions/workflows/cmake.yml/badge.svg?branch=task00&event=push)](https://github.com/GPGPUCourse/GPGPUTasks2025/actions/workflows/cmake.yml)

Установка OpenCL-драйвера для процессора
========================================

Установить OpenCL-драйвер для процессора полезно, даже если у вас есть видеокарта, т.к. на нем удобно тестировать приложение (драйвер видеокарты гораздо чаще может повиснуть вместе с ОС).

Windows
-------

1. Откройте https://software.intel.com/content/www/us/en/develop/tools/opencl-cpu-runtime.html -> ```For Intel CPUs``` (для AMD тоже работает)
2. Скачайте, если у вас нет прокси то с зеркала - [файл w_opencl_runtime_p_2025.2.0.768.exe](https://disk.yandex.ru/d/dlVbMoI3tsPZfw)
3. Установите

Linux (Рекомендуется Ubuntu 24.04)
----------------------------------

Выполните скрипт [install_intel_opencl.sh](/.github/scripts/install_intel_opencl.sh)

Если в процессе запуска этого задания процессор не виден как допустимое OpenCL-устройство - создайте **Issue** в этом репозитории с перечислением:

 - Версия OS
 - Вывод команды ``ls /etc/OpenCL/vendors``
 - Если там в т.ч. есть ``intel64.icd`` файл - то его содержимое (это маленький текстовый файл)

Apple macOS
-----------

Драйвер установлен из коробки, проверьте через ```clinfo``` (можно установить через ```brew```)

Установка OpenCL-драйвера для видеокарты
========================================

Windows
-------

Поставьте драйвер стандартным образом - скачав инсталлятор с официального сайта вендора вашей видеокарты и установив.

Linux
-----

NVidia: я предпочитаю скачивать ```.run```-файл с сайта NVIDIA и устанавливать через него

AMD: [скачав](https://www.amd.com/en/support) и установив amdgpu-pro драйвер

Windows WSL (Ubuntu 24.04)
--------------------------

NVidia: хотя нативной поддержки OpenCL [на данный момент нет](https://github.com/microsoft/WSL/issues/6951) (только CUDA), можно [скомпилировать pocl](https://github.com/microsoft/WSL/issues/6951#issuecomment-2745803886) OpenCL-драйвер который полагается на CUDA.

Apple macOS
-----------

Драйвер установлен из коробки, проверьте через ```clinfo``` (можно установить через ```brew```)

Проверка окружения и начало выполнения задания
==============================================

Про работу под Windows см. в секции [Как работать под windows](#%D0%9A%D0%B0%D0%BA-%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%D1%82%D1%8C-%D0%BF%D0%BE%D0%B4-windows).

1. Сделайте fork этого репозитория
2. ``git clone ВАШ_ФОРК_РЕПОЗИТОРИЯ``
3. ``cd GPGPUTasks2025``
4. ``git checkout task00``
5. ``mkdir build``
6. ``cd build``
7. ``cmake ..``
8. ``make -j8``
9. ``./enumDevices`` должно увидеть хотя бы одну OpenCL-платформу:

```
Number of OpenCL platforms: 1
Platform #1/1
    Platform name: 
```

Если же вы видите ошибку:
```
terminate called after throwing an instance of 'std::runtime_error'
  what(): Can't init OpenCL driver!
Aborted (Core dumped)
```

То попробуйте установить ```sudo apt install ocl-icd-libopencl1``` и выполнить ``./enumDevices`` снова.

Если вы видите ошибку:
```
: CommandLine Error: Option 'polly' registered more than once!
LLVM ERROR: inconsistency in registered CommandLine options
```
То, наоборот, может помочь удалить пакет ```sudo apt remove ocl-icd-libopencl1``` и попробовать выполнить ``./enumDevices`` снова.

Если ``./enumDevices`` не показывает хотя бы одну платформу - создайте **Issue** с перечислением:

 - OS, процессор и видеокарта
 - Успешно ли прошла установка Intel-CPU драйвера
 - Какое было поведение до установки пакета ``ocl-icd-libopencl1`` и какое поведение стало после
 - Вывод ``./enumDevices``

Задание
=======

0. Сделать fork проекта
1. Прочитать все комментарии подряд и выполнить все **TODO** в файле ``src/main.cpp``. Для разработки под Linux рекомендуется использовать CLion. Под Windows рекомендуется использовать CLion+MSVC. Также под Windows можно использовать Visual Studio Community.
2. Отправить **Pull-request** с названием ```Task00 <Имя> <Фамилия> <Аффиляция>```. **Аффиляция** - ваш ВУЗ (например SPbU/ITMO или HSE) или ваше место работы. Мне интересно узнать кто откуда.
3. В тексте **PR** укажите вывод программы при исполнении на сервере Github CI (Github Actions) и на вашем компьютере (в **pre**-тэгах, чтобы сохранить форматирование, см. [пример](https://raw.githubusercontent.com/GPGPUCourse/GPGPUTasks2025/task00/.github/pull_request_example.md)). И ваш бранч должен называться так же, как и у меня - **task00**.
4. Убедиться что Github CI (Github Actions) смог скомпилировать ваш код и что все хорошо, при отправке первого задания CI может не запуститься пока я вручную не нажму ```Approve``` - если я этого не сделал в течение суток - напомните мне пожалуйста в чате курса
5. Ждать комментарии проверки

**Дедлайн**: 23:59 22 сентября. Но убедиться, что хотя бы одно OpenCL-устройство у вас обнаруживается, лучше как можно раньше, чтобы было больше времени на решение проблем если они возникнут (см. **Проверка окружения** выше).

Как работать под Windows
========================

1. Используйте **64-битный компилятор**, т.е. [amd64](/.figures/clion_msvc_settings.png), а не x86. (Если при запуске видите ``Invalid Parameter - 100``, то вы все еще используете 32-битный компилятор)
2. Рекомендуется использовать CLion+MSVC.
3. Можно использовать Visual Studio 2017 Community или новее, она поддерживает CMake-проекты (``File`` -> ``Open`` -> ``Cmake...``). Разве что передавать аргументы запускаемой программе [неудобно](https://docs.microsoft.com/en-us/cpp/ide/cmake-tools-for-visual-cpp?view=vs-2017#configure-cmake-debugging-sessions). Но может быть уже сделали GUI.
