#include <wb.h>
#include <stdio.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY 128

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void gpu_global_queuing_kernel(unsigned int *nodePtrs,
                                          unsigned int *nodeNeighbors,
                                          unsigned int *nodeVisited,
                                          unsigned int *currLevelNodes,
                                          unsigned int *nextLevelNodes,
                                          unsigned int *numCurrLevelNodes,
                                          unsigned int *numNextLevelNodes) {

  //@@ INSERT KERNEL CODE HERE

}

__global__ void gpu_block_queuing_kernel(unsigned int *nodePtrs,
                                         unsigned int *nodeNeighbors,
                                         unsigned int *nodeVisited,
                                         unsigned int *currLevelNodes,
                                         unsigned int *nextLevelNodes,
                                         unsigned int *numCurrLevelNodes,
                                         unsigned int *numNextLevelNodes) {

  //@@ INSERT KERNEL CODE HERE

}

__global__ void gpu_warp_queuing_kernel(unsigned int *nodePtrs,
                                        unsigned int *nodeNeighbors,
                                        unsigned int *nodeVisited,
                                        unsigned int *currLevelNodes,
                                        unsigned int *nextLevelNodes,
                                        unsigned int *numCurrLevelNodes,
                                        unsigned int *numNextLevelNodes) {

  //@@ INSERT KERNEL CODE HERE


}

/******************************************************************************
 Functions
*******************************************************************************/

void cpu_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                 unsigned int *nodeVisited, unsigned int *currLevelNodes,
                 unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
                 unsigned int *numNextLevelNodes) {

  // Loop over all nodes in the curent level
  for (unsigned int idx = 0; idx < *numCurrLevelNodes; ++idx) {
    unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
         ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      if (!nodeVisited[neighbor]) {
        // Mark it and add it to the queue
        nodeVisited[neighbor] = 1;
        nextLevelNodes[*numNextLevelNodes] = neighbor;
        ++(*numNextLevelNodes);
      }
    }
  }
}

void gpu_global_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                        unsigned int *nodeVisited, unsigned int *currLevelNodes,
                        unsigned int *nextLevelNodes,
                        unsigned int *numCurrLevelNodes,
                        unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_block_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                       unsigned int *nodeVisited, unsigned int *currLevelNodes,
                       unsigned int *nextLevelNodes,
                       unsigned int *numCurrLevelNodes,
                       unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_warp_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                      unsigned int *nodeVisited, unsigned int *currLevelNodes,
                      unsigned int *nextLevelNodes,
                      unsigned int *numCurrLevelNodes,
                      unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void setupProblem(const unsigned int numNodes, 
                  unsigned int **nodePtrs_h, 
		  unsigned int **nodeNeighbors_h,
                  unsigned int **nodeVisited_h, 
		  unsigned int **nodeVisited_ref,
                  unsigned int **currLevelNodes_h,
                  unsigned int **nextLevelNodes_h,
                  unsigned int **numCurrLevelNodes_h,
                  unsigned int **numNextLevelNodes_h,
		  const unsigned int *data,
		  const unsigned int data_size) {

  // Initialize node pointers
  *nodePtrs_h = (unsigned int *)malloc((numNodes + 1) * sizeof(unsigned int));
  *nodeVisited_h = (unsigned int *)malloc(numNodes * sizeof(unsigned int));
  *nodeVisited_ref = (unsigned int *)malloc(numNodes * sizeof(unsigned int));
  (*nodePtrs_h)[0] = 0;

  *nodeNeighbors_h =   
      (unsigned int *)malloc(data_size * sizeof(unsigned int));
  
  memset((*nodePtrs_h),0,(numNodes+1)*sizeof(int));
 
  for(unsigned int i = 0; i < data_size; ++i){
    int d_offset = i * 2;
    unsigned int node=data[d_offset];
    (*nodePtrs_h)[node+1]++;
    (*nodeNeighbors_h)[i] = data[d_offset+1];
  }
  
  memset((*nodeVisited_h), 0, numNodes*sizeof(int));
  memset((*nodeVisited_ref), 0, numNodes*sizeof(int));

  unsigned int node_max=(*nodePtrs_h)[0];
  int node_idx=0;
  for(unsigned int node=1;node<numNodes; node++){
    if(node_max<=(*nodePtrs_h)[node + 1]){
      node_max=(*nodePtrs_h)[node + 1];
      node_idx=node;
    }
    (*nodePtrs_h)[node + 1] += (*nodePtrs_h)[node];
    (*nodeVisited_h)[node] = (*nodeVisited_ref)[node] = 0;
  }

  
  *numCurrLevelNodes_h = (unsigned int *)malloc(sizeof(unsigned int));
  **numCurrLevelNodes_h = node_max; // Let level contain 10% of all nodes
  *currLevelNodes_h =
      (unsigned int *)malloc((**numCurrLevelNodes_h) * sizeof(unsigned int));

  unsigned int node_off=(*nodePtrs_h)[node_idx];
  for (unsigned int idx = 0; idx < node_max; ++idx) {
    unsigned int node= (*nodeNeighbors_h)[node_off+idx];
    (*currLevelNodes_h)[idx] = node;
    (*nodeVisited_h)[node] = (*nodeVisited_ref)[node] = 1;
  }

  // Prepare next level containers (i.e. output variables)
  *numNextLevelNodes_h = (unsigned int *)malloc(sizeof(unsigned int));
  **numNextLevelNodes_h = 0;
  *nextLevelNodes_h = (unsigned int *)malloc((numNodes) * sizeof(unsigned int));

}

/////////////////////////////////////////////////////

void zip(const int *a0, const int *a1, size_t n, unsigned int *c) {
  for (size_t i = 0; i < n; ++i) {
     c[(i * 2) + 0] = a0[i];
     c[(i * 2) + 1] = a1[i];
  }
}

int main(int argc, char *argv[]) {
  // Timer timer;

  wbArg_t args;

  args = wbArg_read(argc, argv);
  // Initialize host variables ----------------------------------------------

  // Variables
  unsigned int numNodes;
  unsigned int *nodePtrs_h;
  unsigned int *nodeNeighbors_h;
  unsigned int *nodeVisited_h;
  unsigned int *nodeVisited_ref; // Needed for reference checking
  unsigned int *currLevelNodes_h;
  unsigned int *nextLevelNodes_h;
  unsigned int *numCurrLevelNodes_h;
  unsigned int *numNextLevelNodes_h;
  unsigned int *nodePtrs_d;
  unsigned int *nodeNeighbors_d;
  unsigned int *nodeVisited_d;
  unsigned int *currLevelNodes_d;
  unsigned int *nextLevelNodes_d;
  unsigned int *numCurrLevelNodes_d;
  unsigned int *numNextLevelNodes_d;

  ///////////////////////////////////////////////////////
  unsigned int *data;
  ///////////////////////////////////////////////////////
  
  enum Mode { CPU = 1, GPU_GLOBAL_QUEUE, GPU_BLOCK_QUEUE, GPU_WARP_QUEUE };
  Mode mode;

  int inputLength1;
  int inputLength2;
  int *hostInput1;
  int *hostInput2;

  wbTime_start(Generic, "Importing data and creating memory on host");
  
  mode = (Mode) wbImport_flag(wbArg_getInputFile(args, 0));


  hostInput1 =
      ( int * )wbImport(wbArg_getInputFile(args, 1), &inputLength1, "Integer");
  numNodes = hostInput1[0];
  hostInput1++;
  inputLength1--;

  hostInput2 =
     ( int * )wbImport(wbArg_getInputFile(args, 2), &inputLength2, "Integer");
  (void) hostInput2[0]; // old maxNeighborsPerNode
  hostInput2++;
  inputLength2--; // ignore maxNeighborsPerNode

  assert(inputLength2 == inputLength1);


  wbTime_stop(Generic, "Importing data and creating memory on host"); 

  wbTime_start(Generic, "Setting up the problem...");
  
  data = (unsigned int *)malloc(sizeof(unsigned int)*inputLength1*2);

  zip(hostInput1, hostInput2,  inputLength2 , data);
 

  setupProblem(numNodes, &nodePtrs_h, &nodeNeighbors_h,
               &nodeVisited_h, &nodeVisited_ref, &currLevelNodes_h,
               &nextLevelNodes_h, &numCurrLevelNodes_h, &numNextLevelNodes_h,
	       data,inputLength2);
  

  wbTime_stop(Generic, "Setting up the problem...");

  // Allocate device variables ----------------------------------------------

  if (mode != CPU) {

    wbTime_start(GPU, "Allocating GPU memory.");

    wbCheck(cudaMalloc((void **)&nodePtrs_d,
                       (numNodes + 1) * sizeof(unsigned int)));

    wbCheck(
        cudaMalloc((void **)&nodeVisited_d, numNodes * sizeof(unsigned int)));

    wbCheck(cudaMalloc((void **)&nodeNeighbors_d,
                       nodePtrs_h[numNodes] * sizeof(unsigned int)));

    wbCheck(cudaMalloc((void **)&numCurrLevelNodes_d, sizeof(unsigned int)));

    wbCheck(cudaMalloc((void **)&currLevelNodes_d,
                       (*numCurrLevelNodes_h) * sizeof(unsigned int)));

    wbCheck(cudaMalloc((void **)&numNextLevelNodes_d, sizeof(unsigned int)));

    wbCheck(cudaMalloc((void **)&nextLevelNodes_d,
                       (numNodes) * sizeof(unsigned int)));

    // cudaDeviceSynchronize();
    wbTime_stop(GPU, "Allocating GPU memory.");
  }

  // Copy host variables to device ------------------------------------------

  if (mode != CPU) {
    wbTime_start(GPU, "Copying input memory to the GPU.");

    wbCheck(cudaMemcpy(nodePtrs_d, nodePtrs_h,
                       (numNodes + 1) * sizeof(unsigned int),
                       cudaMemcpyHostToDevice));

    wbCheck(cudaMemcpy(nodeVisited_d, nodeVisited_h,
                       numNodes * sizeof(unsigned int),
                       cudaMemcpyHostToDevice));

    wbCheck(cudaMemcpy(nodeNeighbors_d, nodeNeighbors_h,
                       nodePtrs_h[numNodes] * sizeof(unsigned int),
                       cudaMemcpyHostToDevice));

    wbCheck(cudaMemcpy(numCurrLevelNodes_d, numCurrLevelNodes_h,
                       sizeof(unsigned int), cudaMemcpyHostToDevice));

    wbCheck(cudaMemcpy(currLevelNodes_d, currLevelNodes_h,
                       (*numCurrLevelNodes_h) * sizeof(unsigned int),
                       cudaMemcpyHostToDevice));

    wbCheck(cudaMemset(numNextLevelNodes_d, 0, sizeof(unsigned int)));

    wbTime_stop(GPU, "Copying input memory to the GPU.");
  }

  // Launch kernel ----------------------------------------------------------

  printf("Launching kernel ");

  if (mode == CPU) {
    wbTime_start(Compute, "Performing CPU queuing computation");
    cpu_queuing(nodePtrs_h, nodeNeighbors_h, nodeVisited_h, currLevelNodes_h,
                nextLevelNodes_h, numCurrLevelNodes_h, numNextLevelNodes_h);
    wbTime_stop(Compute, "Performing CPU queuing computation");
  } else if (mode == GPU_GLOBAL_QUEUE) {
    wbTime_start(Compute, "Performing GPU global queuing computation");
    gpu_global_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
                       currLevelNodes_d, nextLevelNodes_d, numCurrLevelNodes_d,
                       numNextLevelNodes_d);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing GPU global queuing computation");
  } else if (mode == GPU_BLOCK_QUEUE) {
    wbTime_start(Compute, "Performing GPU block global queuing computation");
    gpu_block_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
                      currLevelNodes_d, nextLevelNodes_d, numCurrLevelNodes_d,
                      numNextLevelNodes_d);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing GPU block global queuing computation");
  } else if (mode == GPU_WARP_QUEUE) {
    wbTime_start(Compute, "Performing GPU warp global queuing computation");
    gpu_warp_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
                     currLevelNodes_d, nextLevelNodes_d, numCurrLevelNodes_d,
                     numNextLevelNodes_d);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing GPU warp global queuing computation");
  } else {
    // printf("Invalid mode!\n");
    // exit(0);
  }

  // Copy device variables from host ----------------------------------------

  if (mode != CPU) {

    wbTime_start(Copy, "Copying output memory to the CPU");

   wbCheck(cudaMemcpy(numNextLevelNodes_h, numNextLevelNodes_d,
                       sizeof(unsigned int), cudaMemcpyDeviceToHost));
    wbCheck(cudaMemcpy(nextLevelNodes_h, nextLevelNodes_d,
                       numNodes * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));
    wbCheck(cudaMemcpy(nodeVisited_h, nodeVisited_d,
                       numNodes * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));
   
    wbTime_stop(Copy, "Copying output memory to the CPU");
  }

  // Verify correctness -----------------------------------------------------

  wbTime_start(Generic, "Verifying results...");
  wbSolution(args, (int*)nodeVisited_h, numNodes);
  wbTime_stop(Generic, "Verifying results...");

  // Free memory ------------------------------------------------------------

  free(nodePtrs_h);
  free(nodeVisited_h);
  free(nodeVisited_ref);
  free(nodeNeighbors_h);
  free(numCurrLevelNodes_h);
  free(currLevelNodes_h);
  free(numNextLevelNodes_h);
  free(nextLevelNodes_h);
  if (mode != CPU) {
    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(nodePtrs_d);
    cudaFree(nodeVisited_d);
    cudaFree(nodeNeighbors_d);
    cudaFree(numCurrLevelNodes_d);
    cudaFree(currLevelNodes_d);
    cudaFree(numNextLevelNodes_d);
    cudaFree(nextLevelNodes_d);
    wbTime_stop(GPU, "Freeing GPU Memory");
  }

  return 0;
}
