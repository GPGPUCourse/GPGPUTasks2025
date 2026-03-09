# LEADERBOARD

Ниже победители по задачам курса GPGPU 2025.

Замеры делались на `NVIDIA Tesla T4`: `8.1 TFLOPS FP32`, `320 GBytes/s` peak memory bandwidth.

## Task 03. Matrix Multiplication

| Place | Participant | Team | Result | API | Link |
| --- | --- | --- | --- | --- | --- |
| 1 | Роберт Смайт | СПбГУ | 4927.92 GFlops | CUDA, WMMA | [PR #357](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/357) |
| 2 | Mikhail Stulov | МФТИ / ТБанк | 4802.43 GFlops | CUDA, WMMA | [PR #399](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/399) |
| 3 | Хулиган Серега | ИТМО | 4682.12 GFlops | OpenCL | [PR #309](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/309) |

## Task 04. Prefix Sum

| Place | Participant | Team | Result | API | Link |
| --- | --- | --- | --- | --- | --- |
| 1 | Роберт Смайт | СПбГУ | 108.15 GB/s | CUDA | [PR #478](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/478) |
| 2 | Илья Коннов | ИТМО / Яндекс | 92.98 GB/s | OpenCL | [PR #499](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/499) |
| 3 | Mikhail Stulov | МФТИ / ТБанк | 89.48 GB/s | OpenCL | [PR #474](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/474) |

## Task 05. Radix Sort

| Place | Participant | Team | Result | API | Link |
| --- | --- | --- | --- | --- | --- |
| 1 | Роберт Смайт | СПбГУ | 1553 uint millions/s | CUDA | [PR #545](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/545) |
| 2 | Andrew Ratkov | СПбГУ | 451 uint millions/s | OpenCL | [PR #568](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/568) |
| 3 | Pribytkov Fedor | СПбГУ | 257 uint millions/s | OpenCL | [PR #569](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/569) |

## Task 06. Merge Sort

| Place | Participant | Team | Result | API | Link |
| --- | --- | --- | --- | --- | --- |
| 1 | Артём Батыгин | ВШЭ | 655 uint millions/s | OpenCL | [PR #639](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/639) |
| 2 | Mikhail Stulov | МФТИ / ТБанк | 609 uint millions/s | OpenCL | [PR #645](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/645) |
| 3 | Роберт Смайт | СПбГУ | 595 uint millions/s | CUDA | [PR #625](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/625) |

## Task 07. SpMV

| Place | Participant | Team | Result (average over 5 cases) | API | Link |
| --- | --- | --- | --- | --- | --- |
| 1 | Артём Батыгин | ВШЭ | 8722.14 uint millions/s | OpenCL | [PR #666 comment](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/666#issuecomment-3482912238) |
| 2 | Вячеслав Григорович | ИТМО | 8261.34 uint millions/s | OpenCL | [PR #680 comment](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/680#issuecomment-3508555944) |
| 3 | Тяньшэн Цю | ИТМО | 7648.75 uint millions/s | OpenCL | [PR #647 comment](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/647#issuecomment-3478468328) |

## Task 08. Ray Tracing

| Place | Participant | Team | Result | API | Link |
| --- | --- | --- | --- | --- | --- |
| 1 | Тяньшэн Цю | ITMO Team | 40183 coolness = 139 MTris/s x 290 MRays/s | OpenCL | [PR #812](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/812) |
| 2 | Ostapenko Vladislav | HSE Team | 34832 coolness = 165 MTris/s x 212 MRays/s | OpenCL | [PR #803](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/803) |
| 3 | Sanan Kornyakov | HSE Team | 19683 coolness = 83 MTris/s x 238 MRays/s | OpenCL | [PR #813](https://github.com/GPGPUCourse/GPGPUTasks2025/pull/813) |
