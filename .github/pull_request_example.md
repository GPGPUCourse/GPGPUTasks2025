// Убедитесь что название PR соответствует шаблону:
// Task0N <Имя> <Фамилия> <Аффиляция>
// И проверьте что обе ветки PR (отправляемая из вашего форкнутого репозитория и та в которую вы отправляете PR) называются одинаково - task0N

// Впишите сюда (между pre и /pre тэгами) вывод тестирования на вашем компьютере:

<details><summary>Локальный вывод</summary><p>

<pre>
$ ./main_mandelbrot

Found 2 GPUs in 0.05583 sec (OpenCL: 0.0325861 sec, Vulkan: 0.0231783 sec)
Available devices:
  Device #0: API: OpenCL. CPU. 11th Gen Intel(R) Core(TM) i5-1155G7 @ 2.50GHz. Intel(R) Corporation. Total memory: 7692 Mb.
  Device #1: API: OpenCL+Vulkan. GPU. NVIDIA GeForce MX350. Free memory: 1987/1994 Mb.
Using device #0: API: OpenCL. CPU. 11th Gen Intel(R) Core(TM) i5-1155G7 @ 2.50GHz. Intel(R) Corporation. Total memory: 7692 Mb.
Using OpenCL API...
______________________________________________________
Evaluating algorithm #1/3: CPU
algorithm times (in seconds) - 1 values (min=2.97637 10%=2.97637 median=2.97637 90%=2.97637 max=2.97637)
Mandelbrot effective algorithm GFlops: 3.3598 GFlops
saving image to 'mandelbrot CPU.bmp'...
CPU vs CPU average results difference: 0%
______________________________________________________
Evaluating algorithm #2/3: CPU with OpenMP
OpenMP threads: x8 threads
algorithm times (in seconds) - 10 values (min=0.535918 10%=0.550974 median=0.562397 90%=0.581332 max=0.581332)
Mandelbrot effective algorithm GFlops: 17.781 GFlops
saving image to 'mandelbrot CPU with OpenMP.bmp'...
CPU with OpenMP vs CPU average results difference: 0%
______________________________________________________
Evaluating algorithm #3/3: GPU
Kernels compilation done in 0.119127 seconds
algorithm times (in seconds) - 10 values (min=0.03829 10%=0.0386131 median=0.0411772 90%=0.167082 max=0.167082)
Mandelbrot effective algorithm GFlops: 242.853 GFlops
saving image to 'mandelbrot GPU.bmp'...
GPU vs CPU average results difference: 0.942446%


$ ./main_sum

Found 2 GPUs in 0.0547542 sec (OpenCL: 0.0307237 sec, Vulkan: 0.0239773 sec)
Available devices:
  Device #0: API: OpenCL. CPU. 11th Gen Intel(R) Core(TM) i5-1155G7 @ 2.50GHz. Intel(R) Corporation. Total memory: 7692 Mb.
  Device #1: API: OpenCL+Vulkan. GPU. NVIDIA GeForce MX350. Free memory: 1987/1994 Mb.
Using device #0: API: OpenCL. CPU. 11th Gen Intel(R) Core(TM) i5-1155G7 @ 2.50GHz. Intel(R) Corporation. Total memory: 7692 Mb.
Using OpenCL API...
PCI-E bandwidth: 11.4405 GB/s
______________________________________________________
Evaluating algorithm #1/6: CPU
algorithm times (in seconds) - 10 values (min=0.180424 10%=0.180805 median=0.181866 90%=0.188553 max=0.188553)
sum median effective algorithm bandwidth: 2.04837 GB/s
______________________________________________________
Evaluating algorithm #2/6: CPU with OpenMP
algorithm times (in seconds) - 10 values (min=0.0303463 10%=0.0310868 median=0.0402682 90%=0.0476547 max=0.0476547)
sum median effective algorithm bandwidth: 9.2512 GB/s
______________________________________________________
Evaluating algorithm #3/6: 01 atomicAdd from each workItem
Kernels compilation done in 0.100481 seconds
algorithm times (in seconds) - 10 values (min=1.22379 10%=1.22482 median=1.2265 90%=1.32672 max=1.32672)
sum median effective algorithm bandwidth: 0.303733 GB/s
______________________________________________________
Evaluating algorithm #4/6: 02 atomicAdd but each workItem loads K values
Kernels compilation done in 0.0329806 seconds
algorithm times (in seconds) - 10 values (min=0.614065 10%=0.616851 median=0.622302 90%=0.674206 max=0.674206)
sum median effective algorithm bandwidth: 0.59863 GB/s
______________________________________________________
Evaluating algorithm #5/6: 03 local memory and atomicAdd from master thread
Kernels compilation done in 0.049203 seconds
algorithm times (in seconds) - 10 values (min=0.0166418 10%=0.0166913 median=0.0169108 90%=0.0670831 max=0.0670831)
sum median effective algorithm bandwidth: 22.0291 GB/s
______________________________________________________
Evaluating algorithm #6/6: 04 local reduction
Kernels compilation done in 0.0430401 seconds
algorithm times (in seconds) - 10 values (min=0.0182877 10%=0.0183657 median=0.019141 90%=0.0624878 max=0.0624878)
sum median effective algorithm bandwidth: 19.4623 GB/s

</pre>

</p></details>

// Затем создайте PR, должна начать выполняться автоматическиая сборка на Github CI (Github Actions) - рядом с коммитом в PR появится оранжевый шарик (сборка в процессе),
// который потом станет зеленой галкой (прошло успешно) или красным крестиком (что-то пошло не так).
// Затем откройте PR на редактирование чтобы добавить в описание (тоже между pre и /pre тэгами) вывод тестирования на Github CI:
// Чтобы его найти - надо нажать на зеленую галочку или красный крестик рядом с вашим коммитов в рамках PR.
// P.S. В случае если Github CIсборка не запустилась - попробуйте через десять минут или через час добавить фиктивный коммит (например добавив где-то пробел).

<details><summary>Вывод Github CI</summary><p>

<pre>
$ ./main_mandelbrot

Found 2 GPUs in 0.0474998 sec (CUDA: 8.1213e-05 sec, OpenCL: 0.0224659 sec, Vulkan: 0.024905 sec)
Available devices:
  Device #0: API: OpenCL. CPU. AMD EPYC 7763 64-Core Processor                . Intel(R) Corporation. Total memory: 15995 Mb.
  Device #1: API: Vulkan. CPU. llvmpipe (LLVM 20.1.2, 256 bits). Free memory: 15995/15995 Mb.
Using device #0: API: OpenCL. CPU. AMD EPYC 7763 64-Core Processor                . Intel(R) Corporation. Total memory: 15995 Mb.
Using OpenCL API...
______________________________________________________
Evaluating algorithm #1/3: CPU
algorithm times (in seconds) - 1 values (min=1.99944 10%=1.99944 median=1.99944 90%=1.99944 max=1.99944)
Mandelbrot effective algorithm GFlops: 5.00141 GFlops
saving image to 'mandelbrot CPU.bmp'...
CPU vs CPU average results difference: 0%
______________________________________________________
Evaluating algorithm #2/3: CPU with OpenMP
OpenMP threads: x4 threads
algorithm times (in seconds) - 10 values (min=0.602433 10%=0.602449 median=0.608447 90%=0.713998 max=0.713998)
Mandelbrot effective algorithm GFlops: 16.4353 GFlops
saving image to 'mandelbrot CPU with OpenMP.bmp'...
CPU with OpenMP vs CPU average results difference: 0%
______________________________________________________
Evaluating algorithm #3/3: GPU
Kernels compilation done in 0.153135 seconds
algorithm times (in seconds) - 10 values (min=0.149347 10%=0.149392 median=0.149442 90%=0.306597 max=0.306597)
Mandelbrot effective algorithm GFlops: 66.9157 GFlops
saving image to 'mandelbrot GPU.bmp'...
GPU vs CPU average results difference: 0.942446%


$ ./main_sum

Found 2 GPUs in 0.0476257 sec (CUDA: 8.5501e-05 sec, OpenCL: 0.0226199 sec, Vulkan: 0.0248737 sec)
Available devices:
  Device #0: API: OpenCL. CPU. AMD EPYC 7763 64-Core Processor                . Intel(R) Corporation. Total memory: 15995 Mb.
  Device #1: API: Vulkan. CPU. llvmpipe (LLVM 20.1.2, 256 bits). Free memory: 15995/15995 Mb.
Using device #0: API: OpenCL. CPU. AMD EPYC 7763 64-Core Processor                . Intel(R) Corporation. Total memory: 15995 Mb.
Using OpenCL API...
PCI-E bandwidth: 16.3929 GB/s
______________________________________________________
Evaluating algorithm #1/6: CPU
algorithm times (in seconds) - 10 values (min=0.0328594 10%=0.0328849 median=0.0329522 90%=0.0338049 max=0.0338049)
sum median effective algorithm bandwidth: 11.3051 GB/s
______________________________________________________
Evaluating algorithm #2/6: CPU with OpenMP
algorithm times (in seconds) - 10 values (min=0.0213713 10%=0.021422 median=0.0214581 90%=0.0218762 max=0.0218762)
sum median effective algorithm bandwidth: 17.3608 GB/s
______________________________________________________
Evaluating algorithm #3/6: 01 atomicAdd from each workItem
Kernels compilation done in 0.111685 seconds
algorithm times (in seconds) - 10 values (min=1.53749 10%=1.53837 median=1.54221 90%=1.6511 max=1.6511)
sum median effective algorithm bandwidth: 0.241555 GB/s
______________________________________________________
Evaluating algorithm #4/6: 02 atomicAdd but each workItem loads K values
Kernels compilation done in 0.0315077 seconds
algorithm times (in seconds) - 10 values (min=0.770118 10%=0.771143 median=0.772631 90%=0.804455 max=0.804455)
sum median effective algorithm bandwidth: 0.482157 GB/s
______________________________________________________
Evaluating algorithm #5/6: 03 local memory and atomicAdd from master thread
Kernels compilation done in 0.0571756 seconds
algorithm times (in seconds) - 10 values (min=0.0573984 10%=0.0574037 median=0.0576306 90%=0.115326 max=0.115326)
sum median effective algorithm bandwidth: 6.46408 GB/s
______________________________________________________
Evaluating algorithm #6/6: 04 local reduction
Kernels compilation done in 0.0446969 seconds
algorithm times (in seconds) - 10 values (min=0.0584431 10%=0.0584669 median=0.0586625 90%=0.106308 max=0.106308)
sum median effective algorithm bandwidth: 6.35038 GB/s
</pre>

</p></details>
