#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1000000
#define DEBUG 0

int get_occ(char* d_text, char* f_word) {
    int cnt = 0;
    char* p_text = strstr(d_text, f_word);
    while (p_text != NULL) {
        cnt++;
        p_text = strstr(p_text + 1, f_word);
    }
    return cnt;
}

int main(int argc, char* argv[]) {
    // Open the input, encrypted text file
    FILE* inputFile = fopen(argv[1], "r");

    // Open the output, key file
    FILE* keyFile = fopen(argv[2], "w");

    // Open the output, time text file
    FILE* timeFile = fopen(argv[3], "w");

    // Read in the thread count
    char* p1;
    int thread_count = strtol(argv[4], &p1, 10);

    // Allocate and open a buffer to read in the input
    fseek(inputFile, 0L, SEEK_END);
    long lSize = ftell(inputFile);
    rewind(inputFile);
    unsigned char* buffer = calloc(1, lSize + 1);

    // Read the input into the buffer
    fread(buffer, lSize, 1, inputFile);

    int freq = 0;
    int final_key = 0;

    // Record the start time
    double start = omp_get_wtime();

#pragma omp parallel for num_threads(thread_count)
    for (int l_key = 0; l_key < 256; l_key++) {
        unsigned char* d_text = calloc(1, lSize + 1);
        for (int i = 0; i < lSize; i++) {
            int x = buffer[i] - l_key;
            if (x < 0) x += 256;
            d_text[i] = (char)x;
        }

        int l_freq = 0;
        char* f_word = "the";

        char* f_wrd = strtok((char*)d_text, " ");
        while (f_wrd != NULL) {
            if (strcmp(f_wrd, f_word) == 0)
                l_freq++;
            f_wrd = strtok(NULL, " ");
        }
        // printf("%d - %d\n", l_key, freq);
        if (l_freq > freq) {
            freq = l_freq;
            final_key = l_key;
        }
        free(d_text);
    }

    // Record the finish time
    double end = omp_get_wtime();

    // Time calculation (in seconds)
    double time_passed = end - start;

    fprintf(keyFile, "%d", final_key);
    fprintf(timeFile, "%f", time_passed);

    // Cleanup
    fclose(inputFile);
    fclose(keyFile);
    fclose(timeFile);
    free(buffer);

    return 0;
}  // End main //
