# matrix-cuda
matrix multiplication in CUDA, this is a toy program for learning CUDA, some functions are reusable in other project


# test results
#### following tests were carried out on a Tesla M2075 card

[lzhengchun@clus10 liu]$ ./a.out 

please type in m n and k

1024 1024 1024

Time elapsed on matrix multiplication of 1024x1024 . 1024x1024 on GPU: 13.604608 ms.

Time elapsed on matrix multiplication of 1024x1024 . 1024x1024 on CPU: 9925.121094 ms.

all results are correct!!!, speedup = **729.541138**

[lzhengchun@clus10 liu]$ ./a.out 

please type in m n and k

1024 1024 1023

Time elapsed on matrix multiplication of 1024x1024 . 1024x1023 on GPU: 51.141281 ms.

Time elapsed on matrix multiplication of 1024x1024 . 1024x1023 on CPU: 8964.353516 ms.

all results are correct!!!, speedup = **175.286057**

#Notes

(1) function *gpu_matrix_mult*: A naive implementation on GPUs assigns one thread to compute one element of matrix C. Each thread loads one row of matrix A and one column of matrix B from global memory, do the inner product, and store the result back to matrix C in the global memory. In the naive implementation, the amount of computation is 2 x M x N x K flop, while the amount of global memory access is 2 x M x N x K word. The "computation-to-memory ratio" is approximately 1/4 (flop/byte). Therefore, the naive implementation is bandwidth bounded.

(2) function *gpu_square_matrix_mult*: (!!! this is only for square matrix mutiplication)

To increase the "computation-to-memory ratio", the tiled matrix multiplication can be applied. One thread block computes one tile of matrix C. One thread in the thread block computes one element of the tile. The figure shows a 32 x 32 matrix divided into four 16 x 16 tiles. To compute this, four thread blocks each with 16 x 16 threads can be created. The GPU kernel computes C in multiple iterations. In each iteration, one thread block loads one tile of A and one tile of B from global memory to shared memory, performs computation, and stores temporal result of C in register. After all the iteration is done, the thread block stores one tile of C into global memory. For example, a thread block can computer C0,0 in two iterations: C0,0 = A0,0 B0,0 + A0,1 B1,0. Therefore, in the tiled implementation, the amount of computation is still 2 x M x N x K flop. However, using tile size of B, the amount of global memory access is 2 x M x N x K / B word. The "computation-to-memory ratio" is approximately B/4 (flop/byte). We now can tune the "computation-to-memory" ratio by changing the tile size B. Futher explain please redirect to http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmcuda (take care the Pseudocode, some issue was ignored).

As you can see from the test results, tiled version has a much better speedup than *gpu_matrix_mult*. 

#comparison with openmp

(Intel(R) Xeon(R) CPU E5645  @ 2.40GHz) X 4 = 24 Cores

[lzhengchun@clus10 liu]$ ./a.out 

please type in m n and k

2300 2300 2300

Time elapsed on matrix multiplication of 2300x2300 . 2300x2300 on GPU: 166.835617 ms.

Time elapsed on matrix multiplication of 2300x2300 . 2300x2300 on CPU: 19520.644531 ms.

all results are correct!!!, speedup = 117.005257

[lzhengchun@clus10 liu]$ ./a.out 

please type in m n and k

1024 1024 1024

Time elapsed on matrix multiplication of 1024x1024 . 1024x1024 on GPU: 15.479232 ms.

Time elapsed on matrix multiplication of 1024x1024 . 1024x1024 on CPU: 2045.946167 ms.

all results are correct!!!, speedup = **132.173630**

[lzhengchun@clus10 liu]$ ./a.out 

please type in m n and k

1024 1024 1023

Time elapsed on matrix multiplication of 1024x1024 . 1024x1023 on GPU: 53.428638 ms.

Time elapsed on matrix multiplication of 1024x1024 . 1024x1023 on CPU: 1563.460571 ms.

all results are correct!!!, speedup = **29.262594**

So, the openmp version is about 5X faster than single thread version, still far from theoritical (24) 

#todo

(1) further optimization, especially the "computation-to-memory ratio" for non square matrix

(2) solve shared Mem Bank conflict issue and Global Memory does not coalesced issue



