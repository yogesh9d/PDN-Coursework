#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEBUG 0

/* ----------- Project 2 - Problem 1 - Matrix Mult -----------

    This file will multiply two matricies.
    Complete the TODOs in order to complete this program.
    Remember to make it parallelized!
*/
// ------------------------------------------------------ //

int main(int argc, char* argv[]) {
    // Catch console errors
    if (argc != 10) {
        printf("USE LIKE THIS: parallel_mult_mat_mat   mat_1.csv n_row_1 n_col_1   mat_2.csv n_row_2 n_col_2   num_threads   results_matrix.csv   time.csv\n");
        return EXIT_FAILURE;
    }

    // Get the input files
    FILE* inputMatrix1 = fopen(argv[1], "r");
    FILE* inputMatrix2 = fopen(argv[4], "r");

    char* p1;
    char* p2;

    // Get matrix 1's dims
    int n_row1 = strtol(argv[2], &p1, 10);
    int n_col1 = strtol(argv[3], &p2, 10);

    // Get matrix 2's dims
    int n_row2 = strtol(argv[5], &p1, 10);
    int n_col2 = strtol(argv[6], &p2, 10);

    // Get num threads
    int thread_count = strtol(argv[7], NULL, 10);

    // Get output files
    FILE* outputFile = fopen(argv[8], "w");
    FILE* outputTime = fopen(argv[9], "w");

    // TODO: malloc the two input matrices and the output matrix
    // Please use long int as the variable type
    long int* mat1 = (long int*)malloc((n_row1 * n_col1) * sizeof(long int));
    long int* mat2 = (long int*)malloc((n_row2 * n_col2) * sizeof(long int));

    // long int* op = (long int*)malloc((n_row1 * n_col2) * sizeof(long int));

    int max_rc = n_col1;
    if (n_col2 > n_col1)
        max_rc = n_col2;
    int buffsize = max_rc * 7 * sizeof(char);
    char* buff = (char*)malloc(buffsize);
    int cnt = 0;

    // TODO: Parse the input csv files and fill in the input matrices
    for (cnt = 0; feof(inputMatrix1) != true;) {
        fgets(buff, buffsize, inputMatrix1);
        char* tkn = strtok(buff, ", ");
        while (tkn != NULL) {
            mat1[cnt++] = atoi(tkn);
            // printf("%ld ", mat1[cnt-1]);
            tkn = strtok(NULL, ",");
        }
        // printf("\n");
    }

    for (cnt = 0; feof(inputMatrix2) != true;) {
        fgets(buff, buffsize, inputMatrix2);
        char* tkn = strtok(buff, ", ");
        while (tkn != NULL) {
            mat2[cnt++] = atoi(tkn);
            // printf("%ld ", mat2[cnt - 1]);
            tkn = strtok(NULL, ",");
        }
        // printf("\n");
    }

    // We are interesting in timing the matrix-matrix multiplication only
    // Record the start time
    double start = omp_get_wtime();
    long int gmax = -9999999;
    // TODO: Parallelize the matrix-matrix multiplication
#pragma omp parallel for num_threads(thread_count) shared(gmax)
    for (int r1 = 0; r1 < n_row1; r1++) {
        long int lmax = -9999999;
        // #pragma omp parallel for num_threads (thread_count)
        for (int c2 = 0; c2 < n_col2; c2++) {
            // int dest = r1 * c2 + c2;
            //op[r1 * n_col2 + c2] = 0;
            long int val = 0;
            for (int i = 0; i < n_col1; i++)
                val += mat1[r1 * n_col1 + i] * mat2[i * n_col2 + c2];
            if(val > lmax )
                lmax = val;
        }
        if (lmax > gmax)
            gmax = lmax;
    }
    // Record the finish time
    double end = omp_get_wtime();

    // Time calculation (in seconds)
    double time_passed = end - start;

    // Save time to file
    fprintf(outputTime, "%f", time_passed);

    // TODO: save the output matrix to the output csv file
    // for (int r = 0; r < n_row1; r++)
    //     for (int c = 0; c < n_col2; c++)
    //         if ((c != (n_col2 - 1)))
    //             fprintf(outputFile, "%ld, ", op[r * n_col2 + c]);
    //         else
    //             fprintf(outputFile, "%ld\n", op[r * n_col2 + c]);

    fprintf(outputFile, "%ld", gmax);
    // Cleanup
    free(mat1);
    free(mat2);
    //free(op);
    free(buff);

    fclose(inputMatrix1);
    fclose(inputMatrix2);
    fclose(outputFile);
    fclose(outputTime);
    // Remember to free your buffers!

    return 0;
}
