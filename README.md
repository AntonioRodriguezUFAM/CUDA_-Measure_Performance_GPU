# CUDA TIMING

## Timing with clock()
If program uses cudaMemcpy, which is synchronous and waits for previous operations to complete and returns when it is complete, could use clock():

#include <time.h>						// needed for clock()
int main() { 
	clock_t start, stop;					// return types are clock_t, int’s
	…
	start = clock();				// number of clock ticks since prog launched
	cudaMemcpy
	mykernel<<<B,T>>>();				// kernel call
	cudaMemcpy
	stop = clock();
	…
	printf(“Execution time is %f seconds\n", 
                        (float) (stop-start)/(CLOCKS_PER_SEC) ;
	return 0;
}

# Sample partial code to measure performance on GPU
## Using CUDA Events

#define N 1000         // a big number up to INT_MAX, 2,147,483,647
__global__ void gpu_compute(float *result) {   
	int i, j;
	float a = 0.0;
	int tid = blockIdx.x *  blockDim.x + threadIdx.x;

	for (i = 0; i < N; i++) 
	for (j = 0; j < N; j++)  a = a + 0.0001;		// do something, N x N floating pt operations

	result[tid]  = a;	// store result
	return;
}

int main(int argc, char *argv[])  {
	int T = 1, B = 1;            				// threads per block and blocks per grid
	float cpu_result, *gpu_result, ans[T * B];	// result from gpu, to make sure computation is being done

	cudaEvent_t start, end;    				// using cuda events to measure time
	float time;       						// which is applicable for asynchronous code also

	cudaEventCreate(&start);    		 	// instrument code to measure start time
	cudaEventCreate(&end);

	cudaEventRecord(start, 0 );

	cudaMalloc((void**) &gpu_result, T * B * sizeof(float));
	gpu_compute<<<B,T>>>(gpu_result);
	cudaMemcpy(ans,gpu_result, T * B * sizeof(float),cudaMemcpyDeviceToHost);

	cudaEventRecord(end, 0 );    	 		// instrument code to measure end time
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);

	printf("GPU, Answer thread 0, %e\n", ans[0]);
	printf("GPU Number of floating pt operations done %e\n", (double) N * N * T * B);
	printf("GPU Time using CUDA events: %f ms\n", time);  		// time is in ms

	cudaEventDestroy(start);
	cudaEventDestroy(end);
	return 0;
}