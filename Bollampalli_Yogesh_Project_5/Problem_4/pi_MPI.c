#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int my_rank, comm_sz, i, cnt = 0, tot_cnt = 0, n = pow(2, 16);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    double x, y, l_start, l_finish, l_elapsed, pi;

    MPI_Barrier(MPI_COMM_WORLD);
    l_start = MPI_Wtime();

    srand(my_rank + 1);

    for (i = 0; i < n / comm_sz; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            cnt++;
        }
    }

    l_finish = MPI_Wtime();
    l_elapsed = l_finish - l_start;

    MPI_Reduce(&cnt, &tot_cnt, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        FILE* outputFile = fopen(argv[1], "w");
        FILE* timeFile = fopen(argv[2], "w");

        pi = 4.0 * ((double)tot_cnt / (double)n);

        l_finish = MPI_Wtime();
        l_elapsed = l_finish - l_start;
        fprintf(outputFile, "%.10f", pi);
        fprintf(timeFile, "%.10f", l_elapsed);

        fclose(outputFile);
        fclose(timeFile);
    }

    MPI_Finalize();
    return 0;
}
