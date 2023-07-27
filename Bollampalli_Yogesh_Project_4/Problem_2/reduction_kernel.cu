/* Min Kernel --------------------------------------
*       Finds minimum hash value and corresponding nonce value.
*/

#define BLOCK_SIZE 1024

__global__
void reduction_kernel(unsigned int* hash_array, unsigned int* nonce_array, unsigned int array_size) {

    // Calculate thread index
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    // arrays to store the tree reduction performed on hash and nonce array
    __shared__ unsigned int hash_reduction[BLOCK_SIZE];
    __shared__ unsigned int nonce_reduction[BLOCK_SIZE];


    // getting elements from global memory to shared memory
    if (index < array_size){
        hash_reduction[threadIdx.x] = hash_array[index];
        nonce_reduction[threadIdx.x] = nonce_array[index];
    }
    else{
        hash_reduction[threadIdx.x] = UINT_MAX;
        nonce_reduction[threadIdx.x] = UINT_MAX;
    }

    __syncthreads();

    // Tree reduction loop performed on the hash and nonce array
    for (unsigned int stride = blockDim.x /2 ; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
            if(hash_reduction[threadIdx.x + stride] < hash_reduction[threadIdx.x]){
                hash_reduction[threadIdx.x] = hash_reduction[threadIdx.x + stride];
                nonce_reduction[threadIdx.x] = nonce_reduction[threadIdx.x + stride];
            }
        __syncthreads();
    }

    // assigning the reduced values from each block
    if (threadIdx.x == 0) {
        hash_array[blockIdx.x] = hash_reduction[0];
        nonce_array[blockIdx.x] = nonce_reduction[0];
    }

} // End Min Kernel //







