/**
 * Copyright 2013 Erik Zenker, Carlchristian Eckert, Marius Melzer
 *
 * This file is part of HASEonGPU
 *
 * HASEonGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HASEonGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HASEonGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <cstdio>
#include <cassert>
#include <curand_kernel.h>

#include <importance_sampling.hpp>
#include <mesh.hpp>
#include <propagate_ray.hpp>
#include <geometry.hpp>
#include <cudachecks.hpp>
#include <cuda_utils.hpp>
#include <reflection.hpp> /* ReflectionPlane */

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
__global__ void propagateFromTriangleCenter(const Mesh mesh,
					    double *importance,
					    const unsigned sample_i,
					    const double sigmaA,
					    const double sigmaE
					    ){

  double gain = 0;
  unsigned reflection_i = blockIdx.z;
  unsigned reflections = (reflection_i + 1) / 2;
  ReflectionPlane reflectionPlane  = (reflection_i % 2 == 0)? BOTTOM_REFLECTION : TOP_REFLECTION;

  unsigned startPrism = threadIdx.x + blockIdx.x * blockDim.x;
  if(startPrism >= mesh.numberOfPrisms){
    return;
  }
  unsigned startLevel = startPrism/(mesh.numberOfTriangles);
  unsigned startTriangle = startPrism - (mesh.numberOfTriangles * startLevel);
  Point startPoint = mesh.getCenterPoint(startTriangle, startLevel);
  Point samplePoint = mesh.getSamplePoint(sample_i);
  unsigned reflectionOffset = reflection_i * mesh.numberOfPrisms;

  gain = propagateRayWithReflection(startPoint, samplePoint, reflections, reflectionPlane, startLevel, startTriangle, mesh, sigmaA, sigmaE); 
  importance[startPrism + reflectionOffset] = mesh.getBetaVolume(startPrism) * gain;
  if(mesh.getBetaVolume(startPrism) < 0 || gain < 0 || importance[startPrism+reflectionOffset] < 0){
    printf("beta: %f importance: %f gain: %f\n", mesh.getBetaVolume(startPrism), importance[startPrism + reflectionOffset], gain);
  }

}

/**
 * @brief uses a given importance distribution to decide how many rays will be launched from each prism
 *
 * @param *raysDump will contain the number of rays which were mapped to a specific prism
 * 
 * for other parameters, see documentation of importanceSampling()
 */
__global__ void distributeRaysByImportance(Mesh mesh,
					   unsigned *raysPerPrism,
					   double *importance,
					   float *sumPhi,
					   unsigned raysPerSample,
					   unsigned *raysDump){

  unsigned reflection_i = blockIdx.z;
  unsigned reflectionOffset = reflection_i * mesh.numberOfPrisms;

  int startPrism = threadIdx.x + blockIdx.x * blockDim.x;
  if(startPrism >= mesh.numberOfPrisms) return;

  raysPerPrism[startPrism + reflectionOffset] = (unsigned) floor(importance[startPrism + reflectionOffset] / (*sumPhi) * raysPerSample);
  if(raysPerPrism[startPrism + reflectionOffset] > raysPerSample){
	  printf("importance: %f sumPhi: %f raysPerPrism[%d]: %d (max %d)\n",importance[startPrism+reflectionOffset],*sumPhi,startPrism+reflectionOffset,raysPerPrism[startPrism+reflectionOffset],raysPerSample);
  }
  assert(raysPerPrism[startPrism + reflectionOffset] <= raysPerSample);
  atomicAdd(raysDump, raysPerPrism[startPrism + reflectionOffset]);

}

/**
 * @brief takes a number of rays and distributes them randomly over the available prisms
 *        Warning: Does not distribute to reflection slices !!!
 *
 * @param *raysPerPrism the number of rays for each prism (will be changed)
 * @param *raysDump the number of rays which were already distributed
 *
 * for other parameters, see documentation of importanceSampling()
 *
 */
__global__ void distributeRemainingRaysRandomly(Mesh mesh,
						unsigned *raysPerPrism,
						unsigned raysPerSample,
						unsigned *raysDump){
  
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  int raysLeft = raysPerSample - (*raysDump);

  if(id < raysLeft){
    curandState randomState;
    curand_init(id,0,0,&randomState);
    int rand_t = (int ) ceil(curand_uniform(&randomState) * mesh.numberOfTriangles) - 1;
    int rand_z = (int ) ceil(curand_uniform(&randomState) * (mesh.numberOfLevels-1)) - 1;
    unsigned randomPrism = rand_t + rand_z * mesh.numberOfTriangles;
    atomicAdd(&(raysPerPrism[randomPrism]),1);
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
__global__ void recalculateImportance(Mesh mesh,
				      unsigned *raysPerPrism,
				      unsigned raysPerSample,
				      double *importance){


  int startPrism = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned reflection_i = blockIdx.z;
  unsigned reflectionOffset = reflection_i * mesh.numberOfPrisms;

  if(startPrism >= mesh.numberOfPrisms){
    return;
  }
  int startLevel = startPrism/(mesh.numberOfTriangles);
  int startTriangle = startPrism - (mesh.numberOfTriangles * startLevel);

  if(raysPerPrism[startPrism + reflectionOffset] > 0){
    importance[startPrism + reflectionOffset] = raysPerSample * mesh.triangleSurfaces[startTriangle] / (mesh.surfaceTotal * raysPerPrism[startPrism + reflectionOffset]);
  }
  else{
    importance[startPrism + reflectionOffset] = 0;
  }
}

void importanceSamplingPropagation(unsigned sample_i,
			    const unsigned reflectionSlices,
			    Mesh deviceMesh,
			    const double sigmaA,
			    const double sigmaE,
			    double *preImportance,
			    dim3 blockDim,
			    dim3 gridDim){


  dim3 gridDimReflection(gridDim.x, 1, reflectionSlices);
  CUDA_CHECK_KERNEL_SYNC(propagateFromTriangleCenter<<< gridDimReflection, blockDim >>>(deviceMesh, preImportance, sample_i, sigmaA, sigmaE));

}

unsigned importanceSamplingDistribution(
			    const unsigned reflectionSlices,
			    Mesh deviceMesh,
			    const unsigned raysPerSample,
			    double *preImportance,
			    double *importance,
			    unsigned *raysPerPrism,
			    float hSumPhi,
			    const bool distributeRandomly,
			    dim3 blockDim,
			    dim3 gridDim){


  unsigned hRaysDump = 0;

  float *dSumPhi = copyToDevice(hSumPhi);
  unsigned *dRaysDump = copyToDevice(hRaysDump);

  dim3 gridDimReflection(gridDim.x, 1, reflectionSlices);

  CUDA_CHECK_KERNEL_SYNC(distributeRaysByImportance<<< gridDimReflection, blockDim >>>(deviceMesh, raysPerPrism, preImportance, dSumPhi, raysPerSample, dRaysDump));

  // Distribute remaining rays randomly if wanted
  if(distributeRandomly){
    CUDA_CHECK_KERNEL_SYNC(distributeRemainingRaysRandomly<<< 200,blockDim >>>(deviceMesh ,raysPerPrism, raysPerSample, dRaysDump));
    hRaysDump = raysPerSample;
  }
  else {
    hRaysDump = copyFromDevice(dRaysDump);
  }

  CUDA_CHECK_KERNEL_SYNC(recalculateImportance<<< gridDimReflection, blockDim >>>(deviceMesh, raysPerPrism, hRaysDump, importance));

  cudaFree(dSumPhi);
  cudaFree(dRaysDump);

  return hRaysDump;
}
