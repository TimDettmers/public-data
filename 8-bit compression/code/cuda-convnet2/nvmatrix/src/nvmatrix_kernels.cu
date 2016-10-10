/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/nvmatrix_kernels.cuh"

__global__ void kTile(const float* src, float* tgt, const uint srcWidth, const uint srcHeight, const uint tgtWidth, const uint tgtHeight) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    //    const unsigned int numEls = tgtWidth * tgtHeight;
    for (uint i = idx; i < tgtWidth * tgtHeight; i += numThreads) {
        const uint y = i / tgtWidth;
        const uint x = i % tgtWidth;
        const uint srcY = y % srcHeight;
        const uint srcX = x % srcWidth;
        tgt[i] = src[srcY * srcWidth + srcX];
    }
}

__global__ void kDotProduct_r(float* a, float* b, float* target,  const uint numElements) {
    __shared__ float shmem[DP_BLOCKSIZE];

    uint eidx = DP_BLOCKSIZE * blockIdx.x + threadIdx.x;
    shmem[threadIdx.x] = 0;
    if (eidx < gridDim.x * DP_BLOCKSIZE) {
        for (; eidx < numElements; eidx += gridDim.x * DP_BLOCKSIZE) {
            shmem[threadIdx.x] += a[eidx] * b[eidx];
        }
    }
    __syncthreads();
    if (threadIdx.x < 256) {
        shmem[threadIdx.x] += shmem[threadIdx.x + 256];
    }
    __syncthreads();
    if (threadIdx.x < 128) {
        shmem[threadIdx.x] += shmem[threadIdx.x + 128];
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        shmem[threadIdx.x] += shmem[threadIdx.x + 64];
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        volatile float* mysh = &shmem[threadIdx.x];
        *mysh += mysh[32];
        *mysh += mysh[16];
        *mysh += mysh[8];
        *mysh += mysh[4];
        *mysh += mysh[2];
        *mysh += mysh[1];
        if (threadIdx.x == 0) {
            target[blockIdx.x] = *mysh;
        }
    }
}

__global__ void kSetupCurand(curandState *state, unsigned long long seed) {
    const uint tidx = NUM_RND_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, tidx, 0, &state[tidx]);
}


__global__ void kDecompression_8bit(float *flt_tbl, unsigned char *A, float precision, int size, float *out)
{

    const unsigned int numThreads = blockDim.x * gridDim.x;
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    __shared__ float tbl_floats[256];
    if(threadIdx.x < 126)
    {
        tbl_floats[threadIdx.x] = flt_tbl[threadIdx.x]*precision;
        tbl_floats[threadIdx.x+128] = -tbl_floats[threadIdx.x];
    }


    tbl_floats[126] = 0.0f;
    tbl_floats[254] = precision;
    tbl_floats[127] = precision;
    tbl_floats[255] = -precision;

    __syncthreads();

    for (int i = idx;i < size; i += numThreads)
    {
        out[i] = tbl_floats[A[i]];
    }
}

__global__ void kTest(float* out, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += numThreads)
    {
        out[i] = 0.0f;
    }
}

__global__ void kCompression_8bit_standard(float *flt_tbl, float *A, int size, unsigned char *out, float lower, float upper)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float absnumber = 0.0f;
	float threshold_lower = lower;
	float threshold_upper = upper;
	//float threshold_lower = 0.000005;
	//float threshold_upper = 1499;
	//float threshold_lower = 0.00005;
	//float threshold_upper = 14999;
	int isNegative = 0;
	int pivot = 64;
	int upper_pivot = 128;
	int lower_pivot = 0;

	__shared__ float tbl_floats[128];
	if(threadIdx.x < 128)
		tbl_floats[threadIdx.x] = flt_tbl[threadIdx.x];


	__syncthreads();

	  for (int i = idx;i < size; i += numThreads)
	  {
		  isNegative = 0;
		  pivot = 64;
		  upper_pivot = 128;
		  lower_pivot = 0;
		  absnumber = A[i];
		  if(absnumber < 0.0f){isNegative = 1; absnumber=-absnumber; }
		  if(absnumber < threshold_lower){ out[i] = (unsigned char)0; continue; }
		  if(absnumber > threshold_upper){ out[i] = (isNegative == 0 ? (unsigned char)127 : (unsigned char)255); continue; }
		  for(int j = 32; j > 0; j>>=1)
		  {
			  if(absnumber > tbl_floats[pivot])
			  {
				  lower_pivot = pivot;
				  pivot+=j;
			  }
			  else
			  {
				  upper_pivot = pivot;
				  pivot-=j;
			  }

		  }

		  if(lower_pivot == pivot)
			  if(fabsf(tbl_floats[pivot]-absnumber) < (tbl_floats[upper_pivot]-absnumber))
				  if(isNegative == 1)
					  out[i] =  pivot | 1 << 7;
				  else
					  out[i] =  pivot;
			  else
				  if(isNegative == 1)
					  out[i] =  upper_pivot | 1 << 7;
				  else
					  out[i] =  upper_pivot;
		  else
			  if((tbl_floats[pivot]-absnumber) < fabsf(tbl_floats[lower_pivot]-absnumber))
				  if(isNegative == 1)
					  out[i] =  (pivot | 1 << 7);
				  else
					  out[i] =  pivot;
			  else
		  	  	  if(isNegative == 1)
		  	  		  out[i] =  lower_pivot | 1 << 7;
		  		  else
		  			  out[i] =  lower_pivot;

	  }
}

__global__ void kDecompression_8bit_standard(float *flt_tbl, unsigned char *A, int size, float *out)
{

	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	__shared__ float tbl_floats[256];
	if(threadIdx.x < 128)
	{
		tbl_floats[threadIdx.x] = flt_tbl[threadIdx.x];
		tbl_floats[threadIdx.x+128] = -tbl_floats[threadIdx.x];
	}

	__syncthreads();

	for (int i = idx;i < size; i += numThreads)
	{
		out[i] = tbl_floats[A[i]];
	}
}

__global__ void kCompression_8bit_linear(float *flt_tbl, float *A, float precision, int size, unsigned char *out)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	float absnumber = 0.0f;
	float threshold_lower = 0.0000015;
	float threshold_upper = 0.990703;
	int isNegative = 0;
	int pivot = 64;
	int upper_pivot = 128;
	int lower_pivot = 0;

	__shared__ float tbl_floats[128];
	if(threadIdx.x < 128)
		tbl_floats[threadIdx.x] = flt_tbl[threadIdx.x];


	__syncthreads();

	  for (int i = idx;i < size; i += numThreads)
	  {
		  isNegative = 0;
		  pivot = 64;
		  upper_pivot = 128;
		  lower_pivot = 0;
		  absnumber = A[i]/precision;
		  if(absnumber < 0.0f){isNegative = 1; absnumber=-absnumber; }
		  if(absnumber < threshold_lower){ out[i] = (unsigned char)126; continue; }
		  if(absnumber > threshold_upper){ out[i] = (isNegative == 0 ? (unsigned char)127 : (unsigned char)255); continue; }
		  for(int j = 32; j > 0; j>>=1)
		  {
			  if(absnumber > tbl_floats[pivot])
			  {
				  lower_pivot = pivot;
				  pivot+=j;
			  }
			  else
			  {
				  upper_pivot = pivot;
				  pivot-=j;
			  }

		  }

		  if(lower_pivot == pivot)
			  if(fabsf(tbl_floats[pivot]-absnumber) < (tbl_floats[upper_pivot]-absnumber))
				  if(isNegative == 1)
					  out[i] =  pivot | 1 << 7;
				  else
					  out[i] =  pivot;
			  else
				  if(isNegative == 1)
					  out[i] =  upper_pivot | 1 << 7;
				  else
					  out[i] =  upper_pivot;
		  else
			  if((tbl_floats[pivot]-absnumber) < fabsf(tbl_floats[lower_pivot]-absnumber))
				  if(isNegative == 1)
					  out[i] =  (pivot | 1 << 7);
				  else
					  out[i] =  pivot;
			  else
		  	  	  if(isNegative == 1)
		  	  		  out[i] =  lower_pivot | 1 << 7;
		  		  else
		  			  out[i] =  lower_pivot;

	  }
}

__global__ void kCompression_8bit(float *flt_tbl, float *A, float precision, int size, unsigned char *out)
{
    const unsigned int numThreads = blockDim.x * gridDim.x;
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    float absnumber = 0.0f;
    float threshold_lower = 0.0000015;
    float threshold_upper = 0.990703;//991503
    int isNegative = 0;
    int pivot = 63;
    int upper_pivot = 125;
    int lower_pivot = 0;


    __shared__ float tbl_floats[128];
    if(threadIdx.x < 126)
        tbl_floats[threadIdx.x] = flt_tbl[threadIdx.x];


    __syncthreads();

      for (int i = idx;i < size; i += numThreads)
      {
          isNegative = 0;
          pivot = 63;
          upper_pivot = 125;
          lower_pivot = 0;
          absnumber = A[i]/precision;

          if(absnumber < 0.0f){isNegative = 1; absnumber=-absnumber; }
          if(absnumber < threshold_lower){ out[i] = (unsigned char)126; continue; }
          if(absnumber > threshold_upper){ out[i] = (isNegative == 0 ? (unsigned char)127 : (unsigned char)255); continue; }
          for(int j = 32; j > 0; j>>=1)
          {
              if(absnumber > tbl_floats[pivot])
              {
                  lower_pivot = pivot;
                  pivot+=j;
              }
              else
              {
                  upper_pivot = pivot;
                  pivot-=j;
              }

          }


          if(lower_pivot == pivot)
              if(fabsf(tbl_floats[pivot]-absnumber) < (tbl_floats[upper_pivot]-absnumber))
                  if(isNegative == 1)
                      out[i] =  pivot | 1 << 7;
                  else
                      out[i] =  pivot;
              else
                  if(isNegative == 1)
                      out[i] =  upper_pivot | 1 << 7;
                  else
                      out[i] =  upper_pivot;
          else
              if((tbl_floats[pivot]-absnumber) < fabsf(tbl_floats[lower_pivot]-absnumber))
                  if(isNegative == 1)
                      out[i] =  (pivot | 1 << 7);
                  else
                      out[i] =  pivot;
              else
                  if(isNegative == 1)
                      out[i] =  lower_pivot | 1 << 7;
                  else
                      out[i] =  lower_pivot;



      }
}
