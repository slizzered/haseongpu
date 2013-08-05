#include <mesh.h>
#include <stdio.h>
#include <propagate_ray.h>
#include <geometry.h>
#include <assert.h>
#include <curand_kernel.h>
#include <cudachecks.h>

/**
 * @brief calculates a first estimate on the importance of each prism, based on a single ray started in the center of each prism
 *
 * @param *importance will contain the initial importance for each prism
 *
 * @param *sumPhi will contain the cumulative sum of the importance values
 *
 * For other parameters, see documentation of importanceSampling()
 *
 */
__global__ void propagateFromTriangleCenter(
    Mesh mesh,
    double *importance,
    float *sumPhi,
    unsigned sample_i,
    float *sigmaA,
    float *sigmaE,
    double nTot){

  __shared__ double threadPhi[256];
  double gain = 0;
  Ray ray;

  threadPhi[threadIdx.x] = 0;

  int startPrism = threadIdx.x + blockIdx.x * blockDim.x;
  if(startPrism >= mesh.numberOfPrisms){
    return;
  }
  int level_i = startPrism/(mesh.numberOfTriangles);
  unsigned triangle_i = startPrism - (mesh.numberOfTriangles * level_i);
  Point startPoint = mesh.getCenterPoint(triangle_i, level_i);
  Point samplePoint = mesh.getSamplePoint(sample_i);
  unsigned wavelengthOffset = gridDim.y * blockIdx.y;

  ray = generateRay(startPoint, samplePoint);
  gain = propagateRay(ray, level_i, triangle_i, &mesh, sigmaA[blockIdx.y], sigmaE[blockIdx.y], nTot, mesh.thickness);
  importance[startPrism + wavelengthOffset] = mesh.getBetaValue(triangle_i, level_i) * gain;

  threadPhi[threadIdx.x] = importance[startPrism + wavelengthOffset];
  __syncthreads();

  unsigned i = blockDim.x/2;
  while(i != 0){
    if(threadIdx.x < i){
      threadPhi[threadIdx.x] += threadPhi[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }
  if(threadIdx.x == 0){
    atomicAdd(&(sumPhi[blockIdx.y]), float(threadPhi[threadIdx.x]));
  }
}

/**
 * @brief uses a given importance distribution to decide how many rays will be launched from each prism
 *
 * @param *raysDump will contain the number of rays which were mapped to a specific prism
 * 
 * for other parameters, see documentation of importanceSampling()
 */
__global__ void distributeRaysByImportance(
    Mesh mesh,
    unsigned *raysPerPrism,
    double *importance,
    float *sumPhi,
    unsigned raysPerSample,
    unsigned *raysDump){
  __shared__ unsigned raySum[256];
  raySum[threadIdx.x] = 0;
  int startPrism = threadIdx.x + blockIdx.x * blockDim.x;
  if(startPrism >= mesh.numberOfPrisms) return;
  raysPerPrism[startPrism] = (unsigned) floor(importance[startPrism] / (*sumPhi) * raysPerSample);
  raySum[threadIdx.x] = raysPerPrism[startPrism];
  __syncthreads();

  unsigned i = blockDim.x/2;
  while(i != 0){
    if(threadIdx.x < i){
      raySum[threadIdx.x] += raySum[threadIdx.x + i];
    }
    __syncthreads();
    i /= 2;
  }
  if(threadIdx.x == 0){
    atomicAdd(raysDump, raySum[threadIdx.x]);
  }
}

/**
 * @brief takes a number of rays and distributes them randomly over the available prisms
 *
 * @param *raysPerPrism the number of rays for each prism (will be changed)
 *
 * @param *raysDump the number of rays which were already distributed
 *
 * for other parameters, see documentation of importanceSampling()
 *
 */
__global__ void distributeRemainingRaysRandomly(
    Mesh mesh,
    unsigned *raysPerPrism,
    unsigned raysPerSample,
    unsigned *raysDump){

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int raysLeft = raysPerSample-(*raysDump);

  if(id < raysLeft){
    curandState randomState;
    curand_init(id,0,0,&randomState);
    int rand_t = (int ) ceil(curand_uniform(&randomState) * mesh.numberOfTriangles) - 1;
    int rand_z = (int ) ceil(curand_uniform(&randomState) * (mesh.numberOfLevels-1)) - 1;
    atomicAdd(&raysPerPrism[rand_t + rand_z * mesh.numberOfTriangles],1);
  }
}


/**
 * @brief corrects the importance to match with the randomly distributed rays
 *
 * @param *raysPerPrism the number of rays to be launced for each prism
 *
 * @param *importance the importance for each prism (will be changed)
 *
 * for other parameters, see documentation of importanceSampling()
 */
__global__ void recalculateImportance(
    Mesh mesh,
    unsigned *raysPerPrism,
    unsigned raysPerSample,
    double *importance){
  int startPrism = threadIdx.x + blockIdx.x * blockDim.x;
  if(startPrism >= mesh.numberOfPrisms){
    return;
  }
  int startLevel = startPrism/(mesh.numberOfTriangles);
  int startTriangle = startPrism - (mesh.numberOfTriangles * startLevel);
  if(raysPerPrism[startPrism] > 0){
    importance[startPrism] = raysPerSample * mesh.surfaces[startTriangle] / (mesh.surfaceTotal * raysPerPrism[startPrism]);
  }else{
    importance[startPrism] = 0;
  }
}


// unused, because we didn't find a good way to parallelize it...
// OPTIMIZE
// TODO
/**
 * @brief maps every ray to a specific prism
 *
 * @param *raysPerPrism the number of rays to launch in each prism
 *
 * @param raysPerSample the total number of rays to launch 
 *
 * @param *indicesOfPrisms a mapping for each ray to a specific prism
 *
 */
__global__ void createCumulativeSum1(
    Mesh mesh,
    unsigned *raysPerPrism,
    unsigned *cumulativeSums){

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id==0){
    cumulativeSums[0] = 0;
  }
  if(id < mesh.numberOfPrisms-1){
    cumulativeSums[id+1] = raysPerPrism[id];
  }
}

__global__ void createCumulativeSum2(
    Mesh mesh,
    unsigned *cumulativeSums){

  for(int i=0;i<mesh.numberOfPrisms;i++){
    cumulativeSums[i+1]+=cumulativeSums[i];
  }
  //printf("PartialSum sum: %d\n",partialSums[0]);
}

__global__ void mapRaysToPrism(
    Mesh mesh,
    unsigned *raysPerPrism,
    unsigned raysPerSample,
    unsigned *indicesOfPrisms,
    unsigned *cumulativeSums){

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id >= mesh.numberOfPrisms) return;

  unsigned absoluteRay = cumulativeSums[id];
  for(unsigned prism_i=cumulativeSums[id]; prism_i < indicesOfPrisms[id]; ++prism_i){
    for(unsigned ray_i=0; ray_i < raysPerPrism[prism_i]; ++ray_i){
      indicesOfPrisms[absoluteRay++] = prism_i;
    }
  }
}


unsigned importanceSampling(
    unsigned sample_i,
    Mesh deviceMesh,
    unsigned raysPerSample,
    float *sigmaA,
    float *sigmaE,
    double nTot,
    double *importance,
    float *sumPhi,
    unsigned *raysPerPrism,
    unsigned *indicesOfPrisms,
    unsigned *raysDump,
    unsigned *cumulativeSums,
    dim3 threads,
    dim3 blocks){

  float *sumPhiHost = (float*) malloc(blocks.y * sizeof(float));
  unsigned *raysDumpHost = (unsigned*) malloc(blocks.y * sizeof(unsigned));

  for(unsigned i=0; i > blocks.y; ++i){
    sumPhiHost[i] = 0.f;
    raysDumpHost[i] = 0;
  }

 // CUDA_CHECK_RETURN(cudaMemset(sumPhi,sizeof(float),0.f));
  //CUDA_CHECK_RETURN(cudaMemset(raysDump,sizeof(unsigned),0));
  CUDA_CHECK_RETURN(cudaMemcpy(sumPhi,sumPhiHost, blocks.y * sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(raysDump,raysDumpHost, blocks.y * sizeof(unsigned),cudaMemcpyHostToDevice));

  CUDA_CHECK_KERNEL_SYNC(propagateFromTriangleCenter<<< blocks,threads >>>(deviceMesh,importance,sumPhi,sample_i,sigmaA,sigmaE,nTot));
  // TODO wavelength ...
  CUDA_CHECK_KERNEL_SYNC(distributeRaysByImportance<<< blocks,threads >>>(deviceMesh,raysPerPrism,importance,sumPhi,raysPerSample,raysDump));
  CUDA_CHECK_KERNEL_SYNC(distributeRemainingRaysRandomly<<< blocks,threads >>>(deviceMesh,raysPerPrism,raysPerSample,raysDump));
  CUDA_CHECK_KERNEL_SYNC(recalculateImportance<<< blocks, threads >>>(deviceMesh,raysPerPrism,raysPerSample,importance));

//  CUDA_CHECK_KERNEL_SYNC(createCumulativeSum1<<<blocks,threads>>>(deviceMesh, raysPerPrism, cumulativeSums));
//  CUDA_CHECK_KERNEL_SYNC(createCumulativeSum2<<<1,1>>>(deviceMesh, cumulativeSums));
//  CUDA_CHECK_KERNEL_SYNC(mapRaysToPrism<<<blocks,threads>>>(deviceMesh, raysPerPrism, raysPerSample, indicesOfPrisms,cumulativeSums));

  free(sumPhiHost);
  free(raysDumpHost);
  return raysPerSample;
}
