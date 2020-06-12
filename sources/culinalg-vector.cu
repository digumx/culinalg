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

    // Set irepr_ to valid temp state
    irepr_->host_data = nullptr;
    irepr_->device_data = nullptr;
    irepr_->host_data_synced = true;

    // Try to allocate. We keep the local variables void here, unnecessary to do implicit conversion
    void* _h_data, _d_data;
    clg::wrapCudaError<clg::HostAllocationFailedException>(cudaMallocHost(&_h_data, dim_*sizeof(float)));
    clg::wrapCudaError<clg::DeviceAllocationFailedException>(cudaMalloc(&_d_data, dim_*sizeof(float)));

    // Allocation passed
    irepr_->host_data = _h_data;
    irepr_->device_data = _d_data;
}

clg::Vector::~Vector()
{
    // Free data
    cudaFreeHost(irepr_->host_data);
    cudaFree(irepr_->device_data);

    // delte irepr_
    delete irepr_;
}

void clg::copyVector(const Vector& dst, const Vector& src)
{
    // Check dimensionality.
    if(dim_ != other.dim_) throw DimensionalityMismatchException(this->dim_, other.dim_);
    
    // Copy cases based on mem sync situation and record error
    }

clg::Vector::Vector(const Vector& other) : dim_(other.dim_)
{
    // Ensure valid state for this->irepr_
    irepr_->host_data = nullptr;
    irepr_->device_data = nullptr;
    irepr_->host_data_synced = true;

    // Copy data
}
