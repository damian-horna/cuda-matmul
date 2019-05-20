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
\__syncthreads() intrinsic function; \__syncthreads() acts as a barrier at
which all threads in the block must wait before any is allowed to proceed.

## Memory hierarchy

- Each thread has private local memory. Each
thread block has shared memory visible to all threads of the block and with the
same lifetime as the block. All threads have access to the same global memory.

- There are also two additional read-only memory spaces accessible by all threads: the
constant and texture memory spaces. 

## Compute capability
- The compute capability of a device is defined by a major revision number and a minor
revision number.

- Devices with the same major revision number are of the same core architecture. The
major revision number is 3 for devices based on the Kepler architecture, 2 for devices
based on the Fermi architecture, and 1 for devices based on the Tesla architecture.

- The minor revision number corresponds to an incremental improvement to the core
architecture, possibly including new features.

## Hardware implementation

- The CUDA architecture is built around a scalable array of multithreaded Streaming
Multiprocessors (SMs). When a CUDA program on the host CPU invokes a kernel
grid, the blocks of the grid are enumerated and distributed to multiprocessors with
available execution capacity. The threads of a thread block execute concurrently on
one multiprocessor, and multiple thread blocks can execute concurrently on one

- A multiprocessor is designed to execute hundreds of threads concurrently. To
manage such a large amount of threads, it employs a unique architecture called
SIMT (Single-Instruction, Multiple-Thread) that is described in Section 4.1. The
instructions are pipelined to leverage instruction-level parallelism within a single
thread, as well as thread-level parallelism extensively through simultaneous hardware
multithreading as detailed in Section 4.2. Unlike CPU cores they are issued in order
however and there is no branch prediction and no speculative execution.
multiprocessor. As thread blocks terminate, new blocks are launched on the vacated
multiprocessors.

## SIMT architecture

- The multiprocessor creates, manages, schedules, and executes threads in groups of
32 parallel threads called warps. Individual threads composing a warp start together
at the same program address, but they have their own instruction address counter
and register state and are therefore free to branch and execute independently. The
term warp originates from weaving, the first parallel thread technology. A half-warp is
either the first or second half of a warp. A quarter-warp is either the first, second,
third, or fourth quarter of a warp
