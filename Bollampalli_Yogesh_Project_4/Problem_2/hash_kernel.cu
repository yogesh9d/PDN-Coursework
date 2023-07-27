
/* Hash Kernel --------------------------------------
*       Generates an array of hash values from nonces.
*/
#define MAX     123123123

__global__
void hash_kernel(unsigned int* hash_array, unsigned int* nonce_array, unsigned int array_size, unsigned int* transactions, unsigned int n_transactions, unsigned int mod) {

    // Calculate thread index
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    // TODO: Generate hash values
    if (index < array_size){
        hash_array[index] = (nonce_array[index] + transactions[0] * (index + 1)) % MAX;
        for (int j = 1; j < n_transactions; j++)
            hash_array[index] = (hash_array[index] + transactions[j] * (index + 1)) % MAX;
    }

} // End Hash Kernel //
