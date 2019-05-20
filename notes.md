From NVIDIA CUDA C Programming GUIDE: http://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/CUDA_C_Programming_Guide.pdf

## General

- Each block of threads can be scheduled on any of the available
multiprocessors within a GPU, in any order, concurrently or sequentially, so that a
compiled CUDA program can execute on any number of multiprocessors and only the runtime system needs to know the physical
multiprocessor count.

- A GPU is built around an array of Streaming Multiprocessors (SMs).
A multithreaded program is partitioned into blocks of threads that execute independently from each
other, so that a GPU with more multiprocessors will automatically execute the program in less time
than a GPU with fewer multiprocessors.

## Kernels

- CUDA C extends C by allowing the programmer to define C functions, called
kernels, that, when called, are executed N times in parallel by N different CUDA
threads, as opposed to only once like regular C functions.

## Thread hierarchy

- There is a limit to the number of threads per block, since all threads of a block are
expected to reside on the same processor core and must share the limited memory
resources of that core. On current GPUs, a thread block may contain up to 1024
threads.

- However, a kernel can be executed by multiple equally-shaped thread blocks, so that
the total number of threads is equal to the number of threads per block times the
number of blocks.

- Blocks are organized into a one-dimensional, two-dimensional, or three-dimensional
grid of thread blocks as illustrated by Figure 2-1. The number of thread blocks in a
grid is usually dictated by the size of the data being processed or the number of
processors in the system, which it can greatly exceed.

- Thread blocks are required to execute independently: It must be possible to execute
them in any order, in parallel or in series. This independence requirement allows
thread blocks to be scheduled in any order across any number of cores as illustrated
by Figure 1-4, enabling programmers to write code that scales with the number of
cores.

- Threads within a block can cooperate by sharing data through some shared memory
and by synchronizing their execution to coordinate memory accesses. More
precisely, one can specify synchronization points in the kernel by calling the
__syncthreads() intrinsic function; __syncthreads() acts as a barrier at
which all threads in the block must wait before any is allowed to proceed.

## Memory hierarchy
