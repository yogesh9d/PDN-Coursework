#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <driver_types.h>
#include <curand.h>
#include <unistd.h>
#include <cstdio>

#include "support.h"
#include "kernel.cu"

#define BILLION  1000000000.0
#define MAX_LINE_LENGTH 25000

#define BLUR_SIZE 2

#define BLOCK_SIZE 16

void err_check(cudaError_t ret, char* msg, int exit_code);

int main (int argc, char *argv[])
{
    // Check console errors
    if( argc != 6)
    {
        printf("USE LIKE THIS: convolution_serial n_row n_col mat_input.csv mat_output.csv time.csv\n");
        return EXIT_FAILURE;
    }

    // Get dims
    int n_row = strtol(argv[1], NULL, 10);
    int n_col = strtol(argv[2], NULL, 10);

    // Get files to read/write
    FILE* inputFile1 = fopen(argv[3], "r");
    if (inputFile1 == NULL){
        printf("Could not open file %s",argv[2]);
        return EXIT_FAILURE;
    }
    FILE* outputFile = fopen(argv[4], "w");
    FILE* timeFile  = fopen(argv[5], "w");

    // Matrices to use
    int* filterMatrix_h = (int*)malloc(5 * 5 * sizeof(int));
    int* inputMatrix_h  = (int*) malloc(n_row * n_col * sizeof(int));
    int* outputMatrix_h = (int*) malloc(n_row * n_col * sizeof(int));
    int* outputMatrix2_h = (int*) malloc(n_row * n_col * sizeof(int));

    // read the data from the file
    int row_count = 0;
    char line[MAX_LINE_LENGTH] = {0};
    while (fgets(line, MAX_LINE_LENGTH, inputFile1)) {
        if (line[strlen(line) - 1] != '\n') printf("\n");
        char *token;
        const char s[2] = ",";
        token = strtok(line, s);
        int i_col = 0;
        while (token != NULL) {
            inputMatrix_h[row_count*n_col + i_col] = strtol(token, NULL,10 );
            i_col++;
            token = strtok (NULL, s);
        }
        row_count++;
    }


    // Filling filter
	// 1 0 0 0 1
	// 0 1 0 1 0
	// 0 0 1 0 0
	// 0 1 0 1 0
	// 1 0 0 0 1
    for(int i = 0; i< 5; i++)
        for(int j = 0; j< 5; j++)
            filterMatrix_h[i*5+j]=0;

    filterMatrix_h[0*5+0] = 1;
    filterMatrix_h[1*5+1] = 1;
    filterMatrix_h[2*5+2] = 1;
    filterMatrix_h[3*5+3] = 1;
    filterMatrix_h[4*5+4] = 1;

    filterMatrix_h[4*5+0] = 1;
    filterMatrix_h[3*5+1] = 1;
    filterMatrix_h[1*5+3] = 1;
    filterMatrix_h[0*5+4] = 1;

    fclose(inputFile1);


    // --------------------------------------------------------------------------- //
    // ------ Algorithm Start ---------------------------------------------------- //

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    cudaError_t cuda_ret;

    // To use with kernels
    int num_blocks = ceil((float)n_col * n_row / (float)BLOCK_SIZE);
    dim3 dimGrid(ceil((float)n_col/(float) BLOCK_SIZE), ceil((float)n_row/(float) BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    int* device_inputMatrix_h;
    cuda_ret = cudaMalloc((void**)&device_inputMatrix_h, n_row * n_col * sizeof(int));
    err_check(cuda_ret, (char*)"Unable to allocate input matrix to device memory!", 1);
    cuda_ret = cudaMemcpy(device_inputMatrix_h, inputMatrix_h, n_row * n_col * sizeof(int), cudaMemcpyHostToDevice);
    err_check(cuda_ret, (char*)"Unable to read input matrix from host memory!", 3);

    int* device_outputMatrix_h;
    cuda_ret = cudaMalloc((void**)&device_outputMatrix_h, n_row * n_col * sizeof(int));
    err_check(cuda_ret, (char*)"Unable to allocate output matrix to device memory!", 1);

    int* device_filterMatrix_h;
    cuda_ret = cudaMalloc((void**)&device_filterMatrix_h, 5 * 5 * sizeof(int));
    err_check(cuda_ret, (char*)"Unable to allocate filter matrix to device memory!", 1);
    cuda_ret = cudaMemcpy(device_filterMatrix_h, filterMatrix_h, 5 * 5 * sizeof(int), cudaMemcpyHostToDevice);
    err_check(cuda_ret, (char*)"Unable to read filter matrix from host memory!", 3);

    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent1 = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;

    struct timespec kerstart, kerend;
    clock_gettime(CLOCK_REALTIME, &kerstart);

    // Launch the blur kernel
    blur_kernel <<< dimGrid, dimBlock >>> (
        device_inputMatrix_h,
        device_outputMatrix_h,
        device_filterMatrix_h,
        n_row,
        n_col
        );
    cuda_ret = cudaDeviceSynchronize();
    err_check(cuda_ret, (char*)"Unable to launch blur kernel!", 2);

     clock_gettime(CLOCK_REALTIME, &kerend);
    double time_spent2 = (kerend.tv_sec - kerstart.tv_sec) +
                        (kerend.tv_nsec - kerstart.tv_nsec) / BILLION;

    // Get nonces from device memory


    struct timespec devstart, devend;
    clock_gettime(CLOCK_REALTIME, &devstart);

    cuda_ret = cudaMemcpy(outputMatrix_h, device_outputMatrix_h, n_row * n_col * sizeof(int), cudaMemcpyDeviceToHost);
    err_check(cuda_ret, (char*)"Unable to read output matrix from device memory!", 3);



    clock_gettime(CLOCK_REALTIME, &devend);
    double time_spent3 = (devend.tv_sec - devstart.tv_sec) +
                        (devend.tv_nsec - devstart.tv_nsec) / BILLION;
    // --------------------------------------------------------------------------- //
    // ------ Algorithm End ------------------------------------------------------ //

    // --------------------------------------------------------------------------- //
    struct timespec start2, end2;
    clock_gettime(CLOCK_REALTIME, &start2);

    int* device_outputMatrix2_h;
    cuda_ret = cudaMalloc((void**)&device_outputMatrix2_h, n_row * n_col * sizeof(int));
    err_check(cuda_ret, (char*)"Unable to allocate output 2 matrix to device memory!", 1);

    clock_gettime(CLOCK_REALTIME, &end2);
    double time_spent4 = (end2.tv_sec - start2.tv_sec) +
                        (end2.tv_nsec - start2.tv_nsec) / BILLION;

    // Launch the maxpool kernel

    struct timespec kerstart2, kerend2;
    clock_gettime(CLOCK_REALTIME, &kerstart2);

    maxpool_kernel <<< dimGrid, dimBlock >>> (
        device_outputMatrix_h,
        device_outputMatrix2_h,
        n_row,
        n_col
        );
    cuda_ret = cudaDeviceSynchronize();
    err_check(cuda_ret, (char*)"Unable to launch maxpool kernel!", 2);

    clock_gettime(CLOCK_REALTIME, &kerend2);
    double time_spent5 = (kerend2.tv_sec - kerstart2.tv_sec) +
                        (kerend2.tv_nsec - kerstart2.tv_nsec) / BILLION;

    struct timespec devstart2, devend2;
    clock_gettime(CLOCK_REALTIME, &devstart2);

    cuda_ret = cudaMemcpy(outputMatrix2_h, device_outputMatrix2_h, n_row * n_col * sizeof(int), cudaMemcpyDeviceToHost);
    err_check(cuda_ret, (char*)"Unable to read output matrix from device memory!", 3);

    clock_gettime(CLOCK_REALTIME, &devend2);
    double time_spent6 = (devend2.tv_sec - devstart2.tv_sec) +
                        (devend2.tv_nsec - devstart2.tv_nsec) / BILLION;
    // --------------------------------------------------------------------------- //
    // ------ Algorithm End ------------------------------------------------------ //

	// Save output matrix as csv file
    for (int i = 0; i<n_row; i++)
    {
        for (int j = 0; j<n_col; j++)
        {
            fprintf(outputFile, "%d", outputMatrix2_h[i*n_col +j]);
            if (j != n_col -1)
                fprintf(outputFile, ",");
            else if ( i < n_row-1)
                fprintf(outputFile, "\n");
        }
    }

    // Print time
    fprintf(timeFile, "%.20f\n", time_spent1 + time_spent4);
    fprintf(timeFile, "%.20f\n", time_spent2);
    fprintf(timeFile, "%.20f\n", time_spent5);
    fprintf(timeFile, "%.20f\n", time_spent3 + time_spent6);

    // Cleanup
    fclose (outputFile);
    fclose (timeFile);

    free(inputMatrix_h);
    free(outputMatrix_h);
    free(filterMatrix_h);

    return 0;
}

/* Error Check ----------------- //
*   Exits if there is a CUDA error.
*/
void err_check(cudaError_t ret, char* msg, int exit_code) {
    if (ret != cudaSuccess)
        fprintf(stderr, "%s \"%s\".\n", msg, cudaGetErrorString(ret)),
        exit(exit_code);
} // End Error Check ----------- //
