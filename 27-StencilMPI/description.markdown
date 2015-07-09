---
title: Multi-GPU Stencil using MPI
author: PUMPS 2015
---

# Objective
The purpose of this lab is to understand how CUDA manages multiple GPUs and how it interacts with MPI in order to run programs across several nodes. You will optimize a naïve implementation of a 7-point stencil computation using MPI and CUDA, by using streams and page-locked memory to overlap communication and computation.

# Background
The 7-point stencil application is an example of nearest neighbor computations on an 3D input volume. Every element of the output volume is described as a weighted linear combination of the corresponding element and its 6 neighboring values in the input column, as shown in the following figure:

![Stencil Shape](imgs/stencil-shape.png "thumbnail")

In the implementation provided in this lab, each thread block processes a `BLOCK_SIZE` by `BLOCK_SIZE` tile within a for-loop that iterates in the Z-direction. To simplify boundary conditions, the outer planes of all dimensions are initialized to zeros, and the thread blocks process the elements within this boundary. Thus, some of the threads in the blocks that process the outer boundary of the X-Y plane are idle, and the iteration along the Z-direction starts at plane `1` and ends at plane `nz-2`.

In the multi-GPU implementation of this computation, domain decomposition is used to partition data and computation across different GPUs. Since each output point is computed using the nearest neighbors, a data dependency is created between neighboring domains: in order to compute the points in the boundary of a domain, some extra points from the neighboring domains are needed (i.e. halos). This scheme is shown in the following figure:

![Domain decomposition](imgs/stencil-domains.png "thumbnail")

This application executes the stencil kernel several times, using the output of the previous execution as input for the current one. Before starting one interation (i.e. time-step), the halo points must be updated with the new data computed by the neighboring domains in the previous iteration. This can lead to a performance loss since the next iteration cannot proceed until the halos are updated. In order to solve this problem, the points to be transferred can be computed first and the memory transfer can be overlapped with the computation of the rest of the points, as pictured in the following figure.

![Timeline pictorization](imgs/stencil-timeline.png "thumbnail")

This computation scheme requires the utilization of asynchronous data transfers and kernel execution, which in turn require the utilization of CUDA streams and page-locked memory. The main CUDA functions needed to manage these abstractions are:

- `cudaStreamCreate`/`cudaStreamDestroy`: creates/destroys a stream.

- `cudaStreamSynchronize`: blocks until all the operations launched on the stream have finished.

- `cudaMemcpyAsync`: copies data asynchronously (takes an stream as a parameter).

- `cudaHostAlloc`/`cudaFreeHost`: allocates/frees page-locked host memory (neededed to perform asynchronous data transfers between host and device).

# Instructions

In the provided source code, you will find a function named `block2D_stencil`. This function implements the 7-point stencil computation by calling the `block2D_stencil_kernel` kernel. You don't have to modify this code, just call it when necessary.

An initial MPI implementation of the host code is provided as a reference (function `do_stencil`). In this version, first the GPU kernel computes all the points for a domain, and then boundary points are sent and halos are updated with the points received from the neighboring MPI processes. Then, the next iteration can proceed.

You have implement the body of the `do_stencil_overlap` function. It has to do the same as `do_stencil` but using the aforementioned communication/computation overlap scheme to improve the application performance:

- You have to perform as many kernel calls as boundaries in the domain, plus another kernel call to compute the rest of the points. Add offsets to the input/output volumes pointers to control the part of the volume that is processed.

- You have to use CUDA streams and page-locked memory in order to perform asynchronous memory transfers. Use as many streams as needed, remember that operations must be executed on different streams to be overlapped.

- There are 2 datasets available for this lab. Both use the same volume as input data, but dataset `0` executes the provided reference implementation and dataset `1` executes the overlapped implementation that you have to write.

