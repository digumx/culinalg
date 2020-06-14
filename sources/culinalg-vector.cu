/*
 * Implements vector
 */

#include<cassert>

#include<headers/culinalg-vector.hpp>
#include<headers/culinalg-exceptions.hpp>

#include<sources/culinalg-cuheader.cuh>

clg::Vector::alloc_irepr_throw_()
{
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

clg::Vector::

clg::Vector::Vector(size_t n) : dim_(n)
{
    // Make new CuData
    irepr_ = new CuData();

    // Try allocating
    alloc_irepr_throw_();

    // Set to 0 on the CPU
    float* _h_floats = (float*)irepr_->host_data;
    for(size_t i = 0; i < dim_; ++i) _h_floats[i] = 0f;
}

clg::Vector::~Vector()
{
#ifndef NOASSERT
    // This should never be triggered
    if(!irepr_) assert("irepr_ is null!");
#endif

    // Free data
    if(irepr_->host_data)
        cudaFreeHost(irepr_->host_data);
    if(irepr_->device_data)
        cudaFree(irepr_->device_data);

    // delte irepr_
    delete irepr_;
}

clg::Vector::Vector(const Vector& other) : dim_(other.dim_)
{
    // Make new CuData
    irepr_ = new CuData();

    // Ensure valid state for this->irepr_
    irepr_->host_data           = nullptr;
    irepr_->device_data         = nullptr;
    irepr_->host_data_synced    = true;

    // Copy data
    copyCuData(*irepr_, *(other.irepr_), dim_*sizeof(float));
}

clg::Vector::Vector(Vector&& other) : dim_(other.dim_)
{
    // Make new CuData
    irepr_ = new CuData();

    // Just move the data pointers in irepr_
    this->irepr_->host_data         = other.irepr_->host_data;
    this->irepr_->device_data       = other.irepr_->device_data;
    this->irepr_->host_data_synced  = other.irepr_->host_data_synced;

    // Leave other in valid state, not pointing to same data.
    other.irepr_->host_data         = nullptr;
    other.irepr_->device_data       = nullptr;
    other.irepr_->host_data_synced  = true;
}

clg::Vector::operator=(const Vector& other)
{
    // Check dimensionality
    if(dim_ != other.dim_) throw clg::DimensionalityMismatchException(dim_, other.dim_);

    // Attempt to delete any data already present. Almost identical to dtor code, only that it does
    // not have noexcept guarantee. Strong exeption guarantee maintained.
    void* _h_data = irepr_->host_data;
    void* _d_data = irepr_->device_data;
    irepr_->host_data           = nullptr;
    irepr_->device_data         = nullptr;
    irepr_->host_data_synced    = true;
    if(_h_data) 
        clg::wrapCudaError<clg::HostDellocationFailedException>(cudaFreeHost(_h_data));
    if(_d_data)
        clg::wrapCudaError<clg::DeviceDellocationFailedException>(cudaFree(_d_data));

    // Copy data
    copyCuData(*irepr_, *(other.irepr_), dim_*sizeof(float));
}

clg::Vector::operator=(Vector&& other)
{
    // Check dimensionality
    if(dim_ != other.dim_) throw clg::DimensionalityMismatchException(dim_, other.dim_);

}
