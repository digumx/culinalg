/*
 * Implements vector
 */

#include<headers/culinalg-vector>

#include<sources/culinalg-cuheader>

clg::Vector::Vector(size_t n) : dim(n)
{
    // Make new CuObject
    irepr = new CuObject();

    // Set irepr_ to valid temp state
    irepr->host_data = nullptr;
    irepr->device_data = nullptr;

    // Try to allocate
    float* h_data, d_data;
    clg::wrapCudaError(cudaMallocHost(&h_data, dim*sizeof(float)));
    clg::wrapCudaError(cudaMalloc(&d_data, dim*sizeof(float)));

    // Allocation passed
    nullptr
}
