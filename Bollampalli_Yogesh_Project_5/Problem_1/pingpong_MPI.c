#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char* argv[]) {
    // Catch console errors
    if (argc != 3) {
        printf("USE LIKE THIS: pingpong_MPI n_items time_prob1_MPI.csv\n");
        return EXIT_FAILURE;
    }

    /* Read in command line items */
    int n_items = strtol(argv[1], NULL, 10);
    FILE* outputFile = fopen(argv[2], "w");

    /* Start up MPI */
    int* ping_array = (int*)malloc(n_items * sizeof(int));

    int my_rank;
    int comm_sz;


    // TODO: finish setting up MPI
    MPI_Init(NULL, NULL);

    /* Get the number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Get my rank among all the processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // TODO: Create your MPI program.
    if (my_rank == 0) {
        // Fill array with incremental values
        for (int i = 0; i < n_items; i++)
            ping_array[i] = i;

        // Start time
        double starttime;
        starttime = MPI_Wtime();

        // TODO: if myrank is 0
        /* Send message to process 1 and recieve message from process 1 */
        for (int i = 0; i < 1000; i++) {
            MPI_Send(ping_array, n_items, MPI_INT, 1, i, MPI_COMM_WORLD);
            MPI_Recv(ping_array, n_items, MPI_INT, 1, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // End time
        double endtime = MPI_Wtime();

        fprintf(outputFile, "%lf", (endtime - starttime) / 2000);

    } else {
        // TODO: if my rank not 0
        /* Receive message from process 0 and send to process 0*/
        for (int i = 0; i < 1000; i++) {
            MPI_Recv(ping_array, n_items, MPI_INT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(ping_array, n_items, MPI_INT, 0, i, MPI_COMM_WORLD);
        }
    }

    free(ping_array);
    fclose(outputFile);

    /* Shut down MPI */
    MPI_Finalize();

    return 0;
} /* main */

