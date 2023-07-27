#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>  // for strtol
#include <string.h>
#include <time.h>

#define MAXCHAR 25
#define BILLION 1000000000.0

int main(int argc, char* argv[]) {

    int vec_size, lvec_size, comm_sz, my_rank;
    double *vec_1, *vec_2, ldot_product = 0.0, dot_product = 0.0;
    struct timespec start, end;
    FILE *outputFile, *timeFile;

    // TODO: finish setting up MPI
    MPI_Init(NULL, NULL);

    /* Get the number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Get my rank among all the processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        if (argc != 6) {
            printf("USE LIKE THIS: dotprod_serial vector_size vec_1.csv vec_2.csv result.csv time.csv\n");
            return EXIT_FAILURE;
        }

        // Size
        vec_size = strtol(argv[1], NULL, 10);

        // Input files
        FILE* inputFile1 = fopen(argv[2], "r");
        FILE* inputFile2 = fopen(argv[3], "r");
        if (inputFile1 == NULL) printf("Could not open file %s", argv[2]);
        if (inputFile2 == NULL) printf("Could not open file %s", argv[3]);

        // Output files
        outputFile = fopen(argv[4], "w");
        timeFile = fopen(argv[5], "w");

        // To read in
        vec_1 = (double*)malloc(vec_size * sizeof(double));
        vec_2 = (double*)malloc(vec_size * sizeof(double));

        // Store values of vector
        int k = 0;
        char str[MAXCHAR];
        while (fgets(str, MAXCHAR, inputFile1) != NULL) {
            sscanf(str, "%lf", &(vec_1[k]));
            k++;
        }
        fclose(inputFile1);

        // Store values of vector
        k = 0;
        while (fgets(str, MAXCHAR, inputFile2) != NULL) {
            sscanf(str, "%lf", &(vec_2[k]));
            k++;
        }
        fclose(inputFile2);

        // Get start time
        clock_gettime(CLOCK_REALTIME, &start);
    }

    // Get dot product
    MPI_Bcast(&vec_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    lvec_size = vec_size / comm_sz;

    double* lvec_1 = (double*)malloc(lvec_size * sizeof(double));
    double* lvec_2 = (double*)malloc(lvec_size * sizeof(double));

    MPI_Scatter(vec_1, lvec_size, MPI_DOUBLE, lvec_1, lvec_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(vec_2, lvec_size, MPI_DOUBLE, lvec_2, lvec_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < vec_size; i++)
        ldot_product += lvec_1[i] * lvec_2[i];

    MPI_Reduce(&ldot_product, &dot_product, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        // Get finish time
        clock_gettime(CLOCK_REALTIME, &end);
        double time_spent = (end.tv_sec - start.tv_sec) +
                            (end.tv_nsec - start.tv_nsec) / BILLION;

        // Print to output
        fprintf(outputFile, "%lf", dot_product);
        fprintf(timeFile, "%.20f", time_spent);

        // Cleanup
        fclose(outputFile);
        fclose(timeFile);
    }

    free(vec_1);
    free(vec_2);

    return 0;
}
