/*
 * Implements vector
 */

#include<headers/culinalg-vector.hpp>
#include<headers/culinalg-exceptions.hpp>

#include<sources/culinalg-cuheader.cuh>

clg::Vector::Vector(size_t n) : dim_(n)
{
    // Make new CuObject
    irepr_ = new CuObject();

    // Set irepr__ to valid temp state
    irepr_->host_data = nullptr;
    irepr_->device_data = nullptr;

    // Try to allocate
    float* _h_data, _d_data;
    clg::wrapCudaError<clg::HostAllocationFailedException>(cudaMallocHost(&_h_data, dim_*sizeof(float)));
    clg::wrapCudaError<clg::DeviceAllocationFailedException>(cudaMalloc(&_d_data, dim_*sizeof(float)));

    // Allocation passed
    irepr_->host_data = _h_data;
    irepr_->device_data = _d_data;
}

clg::Vector::~Vector()
{
    cudaFreeHost(irepr_->host_data);
    cudaFree(irepr_->device_data);
}

// TODO complete
//clg::Vector::Vector(const Vector& other)
//{
//    // Check dimensionality.
//    if(dim_ != other.dim_) throw DimensionalityMismatchException(this->dim_, other.dim_); 
//}
