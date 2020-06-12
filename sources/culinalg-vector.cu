/*
 * Implements vector
 */

#include<headers/culinalg-vector.hpp>
#include<headers/culinalg-exceptions.hpp>

#include<sources/culinalg-cuheader.cuh>

clg::Vector::Vector(size_t n) : dim(n)
{
    // Make new CuObject
    irepr = new CuObject();

    // Set irepr_ to valid temp state
    irepr->host_data = nullptr;
    irepr->device_data = nullptr;

    // Try to allocate
    float* h_data, d_data;
    clg::wrapCudaError<clg::HostAllocationFailedException>(cudaMallocHost(&h_data, dim*sizeof(float)));
    clg::wrapCudaError<clg::DeviceAllocationFailedException>(cudaMalloc(&d_data, dim*sizeof(float)));

    // Allocation passed
    irepr->host_data = h_data;
    irepr->device_data = h_data;
}

clg::Vector::~Vector()
{
    // Copy over refs and nullify them
    float* h_data = irepr->host_data;
    float* d_data = irepr->device_data;
    irepr->host_data = nullptr;
    irepr->device_data = nullptr;

    // Free memory. Do not worry about exceptions, as this is a destructor and there isn't much we
    // can do.
    cudaFreeHost(h_data);
    cudaFree(h_data);
}
