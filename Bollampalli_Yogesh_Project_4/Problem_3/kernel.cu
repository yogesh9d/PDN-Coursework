#define BLUR_SIZE 2

__global__ void blur_kernel(int* input, int* output, int* filter, int n_row, int n_col)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_row || j >= n_col) return;
    int sum_val = 0;
    for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE +1; ++blurRow)
    {
        for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol)
        {
            int curRow = i + blurRow;
            int curCol = j + blurCol;
            int i_row = blurRow + BLUR_SIZE;
            int i_col = blurCol + BLUR_SIZE;
            if( curRow > -1 && curRow < n_row && curCol > -1 && curCol < n_col)
            {
                sum_val += input[curRow*n_col + curCol]*filter[i_row*5 + i_col];
            }
        }
    }
    output[i*n_col+j] = sum_val;
}
