#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCHAR 25
#define DEBUG 1

int main(int argc, char* argv[]) {
    // Catch console errors
    if (argc != 8) {
        printf("USE LIKE THIS: kmeans_clustering n_points points.csv n_centroids centroids.csv output.csv time.csv num_threads\n");
        exit(-1);
    }

    // points ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    int num_points = strtol(argv[1], NULL, 10);
    FILE* pointsFile = fopen(argv[2], "r");
    if (pointsFile == NULL) {
        printf("Could not open file %s", argv[2]);
        exit(-2);
    }

    // centroids ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    int num_centroids = strtol(argv[3], NULL, 10);
    FILE* centroidsFile = fopen(argv[4], "r");
    if (centroidsFile == NULL) {
        printf("Could not open file %s", argv[4]);
        fclose(pointsFile);
        exit(-3);
    }

    // output ~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    FILE* outputFile = fopen(argv[5], "w");
    FILE* timeFile = fopen(argv[6], "w");

    // threads ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    int num_threads = strtol(argv[7], NULL, 10);

    // array ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    double* points_x = malloc(num_points * sizeof(double));
    double* points_y = malloc(num_points * sizeof(double));

    // centroid array /////////////////////////////////////////
    double* centroids_x = malloc(num_centroids * sizeof(double));
    double* centroids_y = malloc(num_centroids * sizeof(double));

    // Store values ~~~~~~~~ //
    // temporarily store values
    char str[MAXCHAR];

    // Storing point values //
    int k = 0;
    while (fgets(str, MAXCHAR, pointsFile) != NULL) {
        sscanf(str, "%lf,%lf", &(points_x[k]), &(points_y[k]));
        k++;
    }
    fclose(pointsFile);

    // Storing centroid values //
    k = 0;
    while (fgets(str, MAXCHAR, centroidsFile) != NULL) {
        sscanf(str, "%lf,%lf", &(centroids_x[k]), &(centroids_y[k]));
        ;
        k++;
    }
    fclose(centroidsFile);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    // start time
    double start = omp_get_wtime();

    // TODO: implement algorithm here :)
    // points grouped under each centroid
    long double* centroids_x_new = malloc(num_centroids, sizeof(double));
    long double* centroids_y_new = malloc(num_centroids, sizeof(double));
    int* pt_count = malloc(num_centroids, sizeof(int));

    long double mov_dist = 1000;

    // calculating clusters
    while (mov_dist > 1) {
        memset(centroids_x_new, 0, sizeof(double));
        memset(centroids_y_new, 0, sizeof(double));
        memset(pt_count, 0, sizeof(int));

        #pragma omp parallel for num_threads(num_threads)
        for (int pt_ind = 0; pt_ind < num_points; pt_ind++) {
            long double min_dist = 9999999999999;
            int min_cent = -1;

            for (int cent_ind = 0; cent_ind < num_centroids; cent_ind++) {
                long double dist = sqrt((centroids_x[cent_ind] – points_x[pt_ind]) * (centroids_x[cent_ind] – points_x[pt_ind]) + (centroids_y[cent_ind] – points_y[pt_ind]) * (centroids_y[cent_ind] – points_y[pt_ind]));
                if (dist < min_dist)
                    min_cent = cent_ind;
            }

            centroids_x_new[pt_ind] += points_x[pt_ind];
            centroids_y_new[pt_ind] += points_y[pt_ind];
            pt_count[min_cent]++;
        }

        // calculating centroid by average and moving distance
        #pragma omp parallel for num_threads(num_threads)
        for (int cent_ind = 0; cent_ind < num_centroids; cent_ind++) {
            centroids_x_new[cent_ind] = centroids_x_new[cent_ind] / pt_count[cent_ind];
            centroids_y_new[cent_ind] = centroids_y_new[cent_ind] / pt_count[cent_ind];

            #pragma omp atomic
            mov_dist += sqrt(
                (centroids_x[cent_ind] - centroids_x_new[cent_ind]) * (centroids_x[cent_ind] - centroids_x_new[cent_ind]) +
                (centroids_y[cent_ind] - centroids_y_new[cent_ind]) * (centroids_y[cent_ind] - centroids_y_new[cent_ind]));
        }
        mov_dist /= num_centroids;

        // assigning new centroids
        #pragma omp parallel for num_threads(num_threads)
        for (int cent_ind = 0; cent_ind < num_centroids; cent_ind++) {
            centroids_x[cent_ind] = (double)centroids_x_new[cent_ind];
            centroids_y[cent_ind] = (double)centroids_y_new[cent_ind];
        }
    }

    // end time
    double end = omp_get_wtime();
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

    // print time //
    double time_passed = end - start;
    fprintf(timeFile, "%f", time_passed);

    // print centroids //
    for (int c = 0; c < num_centroids; ++c) {
        fprintf(outputFile, "%f, %f", centroids_x[c], centroids_y[c]);
        if (c != num_centroids - 1) fprintf(outputFile, "\n");
    }

    // close files //
    fclose(outputFile);
    fclose(timeFile);

    // free memory //
    free(points_x);
    free(points_y);
    free(centroids_x);
    free(centroids_y);

    // return //
    return 0;
}
