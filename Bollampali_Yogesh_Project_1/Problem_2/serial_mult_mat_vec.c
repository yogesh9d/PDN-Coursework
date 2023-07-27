#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Use more libraries as necessary

#define DEBUG 0

/* ---------- Project 1 - Problem 2 - Mat-Vec Mult ----------
    This file will multiply a matrix and vector.
    Complete the TODOs left in this file.
*/
// ------------------------------------------------------ //

int main(int argc, char* argv[]) {
    // Catch console errors
    if( argc != 7)
    {
        printf("USE LIKE THIS: serial_mult_mat_vec in_mat.csv n_row_1 n_col_1 in_vec.csv n_row_2 output_file.csv \n");
        return EXIT_FAILURE;
    }

    // Get the input files
    FILE *matFile = fopen(argv[1], "r");
    FILE *vecFile = fopen(argv[4], "r");

    // Get dim of the matrix
    char* p1;
    char* p2;
    int n_row1 = strtol(argv[2], &p1, 10 );
    int n_col1 = strtol(argv[3], &p2, 10 );

    // Get dim of the vector
    char* p3;
    int n_row2 = strtol(argv[5], &p3, 10 );

    // Get the output file
    FILE *outputFile = fopen(argv[6], "w");

    // TODO: Use malloc to allocate memory for the matrices
    long int* mat = (long int*)malloc((n_row1 * n_col1) * sizeof(long int));
    long int* vect = (long int*)malloc((n_row2) * sizeof(long int));
    long int* op = (long int*)malloc((n_row2) * sizeof(long int));
    int k = n_row1;
    if (n_col1 > n_row1)
        k = n_col1;
    int buffsize = k * 7 * sizeof(char);
    char* buff = (char*)malloc(buffsize);
    int cnt = 0;

    // TODO: Parse the input CSV files
    for (cnt = 0; feof(matFile) != true;) {
        fgets(buff, buffsize, matFile);
        char* tkn = strtok(buff, ", ");
        while (tkn != NULL) {
            mat[cnt++] = atoi(tkn);
            // printf("%ld ", mat[cnt-1]);
            tkn = strtok(NULL, ",");
        }
        // printf("\n");
    }

    for (cnt = 0; feof(vecFile) != true;) {
        fgets(buff, buffsize, vecFile);
        char* tkn = strtok(buff, ", ");
        while (tkn != NULL) {
            vect[cnt++] = atoi(tkn);
            // printf("%ld ", vect[cnt - 1]);
            tkn = strtok(NULL, ",");
        }
        // printf("\n");
    }
    // TODO: Perform the matrix-vector multiplication
    for (int cnt = 0, i = 0, v = 0; i < n_row1; i++, cnt++) {
        op[cnt] = 0;
        for (int j = 0; j < n_col1; j++)
            op[cnt] += vect[j] * mat[(n_col1 * i) + j];
    }
    // TODO: Write the output CSV file
    for (int i = 0; i < n_row1; i++)
        if ((i < (n_row1 - 1)))
            fprintf(outputFile, "%ld\n", op[i]);
        else
            fprintf(outputFile, "%ld", op[i]);
    // TODO: Free memory
    free(mat);
    free(vect);
    free(op);
    free(buff);
    // Cleanup
    fclose(matFile);
    fclose(vecFile);
    fclose(outputFile);
    // Free buffers here as well!

    return 0;
}
