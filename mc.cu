#include<time.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<cuda.h>
#include<cuda_runtime.h>

#define NTPB 128            /* Number of Threads Per Block */


__device__ inline void myAtomicAdd(double *address, double value)  //See CUDA official forum
 {
    unsigned long long oldval, newval, readback;
 
    oldval = __double_as_longlong(*address);
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
    {
        oldval = readback;
        newval = __double_as_longlong(__longlong_as_double(oldval) + value);
    }
 }

__global__ void integrater(float *x, float *y, float *z, float *u, float *v, float *I1, int n){
    int i;
    int iglob = threadIdx.x + blockIdx.x*blockDim.x; 
    int iloc  = threadIdx.x                        ;
    extern __shared__ float block_cache[]; 

    if (iglob < n)
	block_cache[iloc] = expf(-x[iglob]*x[iglob] - y[iglob]*y[iglob]-z[iglob]*z[iglob]-v[iglob]*v[iglob]-u[iglob]*u[iglob]);/*main function eval*/
    else
	block_cache[iloc] = 0;

    __syncthreads();

    /* on the "master thread" of each block" sum the pairwise products
       on that block into the block's portion of the global sum */
    if (iloc == 0){
	float sum = 0.0;
	for (i=0;i<NTPB;++i)
	    sum += block_cache[i];
	atomicAdd(I1,sum);  
    }

}


int main(int argc, char **argv){
    float *x,   *y,   *z, *u, *v, *I1;       /* host pointers */
    float *x_d, *y_d, *z_d, *u_d, *v_d, *I1_d;     /* device pointers */
    int i,n;                  /* vector length */
    cudaEvent_t start, stop;  /* timers */
    float times;
    float actual = .232322;

    n = atoi(argv[1]);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* allocate host memory */
    assert (cudaMallocHost((void **) &x, n*sizeof(float)) == cudaSuccess);
    assert (cudaMallocHost((void **) &y, n*sizeof(float)) == cudaSuccess);
    assert (cudaMallocHost((void **) &z, n*sizeof(float)) == cudaSuccess);
    assert (cudaMallocHost((void **) &u, n*sizeof(float)) == cudaSuccess);
    assert (cudaMallocHost((void **) &v, n*sizeof(float)) == cudaSuccess);
    assert (cudaMallocHost((void **) &I1, 1*sizeof(float)) == cudaSuccess);

    srand((time(NULL)));
    for (i=0;i<n;++i){
	x[i] = (float)rand()/(float)(RAND_MAX-1);
	y[i] = (float)rand()/(float)(RAND_MAX-1);
	z[i] = (float)rand()/(float)(RAND_MAX-1);
	u[i] = (float)rand()/(float)(RAND_MAX-1);
	v[i] = (float)rand()/(float)(RAND_MAX-1);
	
    }

    *I1 = 0.0;
    /* allocate memory on device */
    assert (cudaMalloc((void **) &x_d, n*sizeof(float)) == cudaSuccess);
    assert (cudaMalloc((void **) &y_d, n*sizeof(float)) == cudaSuccess);
    assert (cudaMalloc((void **) &z_d, n*sizeof(float)) == cudaSuccess);
    assert (cudaMalloc((void **) &u_d, n*sizeof(float)) == cudaSuccess);
    assert (cudaMalloc((void **) &v_d, n*sizeof(float)) == cudaSuccess);
    assert (cudaMalloc((void **) &I1_d, 1*sizeof(float)) == cudaSuccess);

    /* copy host data to device pointers */
    assert(cudaMemcpy(x_d,x,n*sizeof(float),cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(y_d,y,n*sizeof(float),cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(z_d,z,n*sizeof(float),cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(u_d,u,n*sizeof(float),cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(v_d,v,n*sizeof(float),cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(I1_d,I1,1*sizeof(float),cudaMemcpyHostToDevice) == cudaSuccess);


    /* launch and time kernel code */
    cudaEventRecord( start, 0 );  

    integrater<<<(n+NTPB-1)/NTPB,NTPB,NTPB*sizeof(float)>>>(x_d,y_d, z_d, u_d, v_d, I1_d,n);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &times, start, stop );

    assert(cudaMemcpy(I1,I1_d,1*sizeof(float),cudaMemcpyDeviceToHost) == cudaSuccess);
    *I1 = *I1/(float)n;
    printf("value: %f\nerror: %f\ntime elapsed: %f(s)\n", *I1, fabs(*I1-actual)/actual, times);
    cudaFree(x_d);  cudaFree(y_d);  cudaFree(z_d);
    cudaFree(I1_d);  cudaFree(u_d);  cudaFree(v_d);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );


}

