#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXLINE 25
#define DEBUG 0

// to read in file
/* Read Input -------------------- */
float* read_input(FILE* inputFile, int n_items) {
    float* arr = (float*)malloc(n_items * sizeof(float));
    char line[MAXLINE] = {0};
    int i = 0;
    while (fgets(line, MAXLINE, inputFile)) {
        sscanf(line, "%f", &(arr[i]));
        ++i;
    }
    return arr;
}  // Read Input //

/* Cmp Int ----------------------------- */
// use this for qsort
// source: https://stackoverflow.com/questions/3886446/problem-trying-to-use-the-c-qsort-function
int cmpfloat(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}  // Cmp Int //

void combine(float* first_arr, int len1, float* second_arr, int len2, float* result_arr) {
    int pos1 = 0, pos2 = 0, res_pos = 0;

    while (pos1 < len1 && pos2 < len2) {
        if (first_arr[pos1] <= second_arr[pos2]) {
            result_arr[res_pos++] = first_arr[pos1++];
        } else {
            result_arr[res_pos++] = second_arr[pos2++];
        }
    }

    for (; pos1 < len1; pos1++) {
        result_arr[res_pos++] = first_arr[pos1];
    }

    for (; pos2 < len2; pos2++) {
        result_arr[res_pos++] = second_arr[pos2];
    }
}

/* Main Program -------------- */
int main(int argc, char* argv[]) {
    // Start MPI
    int my_rank, comm_sz, l_size, n_items = 0;
    float *g_arr, *l_arr, *res;
    double elapsed;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // arrays to use
    // TODO: initialize your arrays here

    if (my_rank == 0) {
        if (argc != 5) {
            printf("USE LIKE THIS: merge_sort_MPI n_items input.csv output.csv time.csv\n");
            return EXIT_FAILURE;
        }

        // input file and size
        FILE* inputFile = fopen(argv[2], "r");
        n_items = strtol(argv[1], NULL, 10);

        l_size = n_items / comm_sz;

        // To read in
        g_arr = (float*)malloc(n_items * sizeof(float));

        // Store values of g_arr
        memcpy(g_arr, read_input(inputFile, n_items), n_items * sizeof(float));
    }

    l_arr = (float*)malloc(l_size * sizeof(float));

    // get start time double local_start, local_finish, local_elapsed, elapsed;
    MPI_Barrier(MPI_COMM_WORLD);
    double local_start = MPI_Wtime();

    // TODO: implement solution here
    MPI_Scatter(g_arr, l_size, MPI_FLOAT, l_arr, l_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    qsort(l_arr, l_size, sizeof(float), cmpfloat);

    MPI_Gather(l_arr, l_size, MPI_FLOAT, g_arr, l_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // get elapsed time
    double local_finish = MPI_Wtime();
    double local_elapsed = local_finish - local_start;

    // send time to main process
    MPI_Reduce(
        &local_elapsed,
        &elapsed,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD);

    // Write output (Step 5)
    if (my_rank == 0) {
        res = (float*)malloc(n_items * sizeof(float));
        memcpy(res, g_arr, l_size * sizeof(float));
        for (int i = 1; i < comm_sz; i++)
            combine(res, l_size * i, g_arr + l_size * i, l_size, res);

        FILE* outputFile = fopen(argv[3], "w");
        FILE* timeFile = fopen(argv[4], "w");

        // TODO: output
        // Print to output
        for (int i = 0; i < n_items; i++) {
            fprintf(outputFile, "%.6f\n", res[i]);
        }

        fprintf(timeFile, "%.20f", elapsed);

        fclose(outputFile);
        fclose(timeFile);
    }

    MPI_Finalize();

    if (DEBUG) printf("Finished!\n");
    return 0;

}  // End Main //
