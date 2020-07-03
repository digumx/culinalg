/*
 * Implements vector
 */

#include<cassert>
#include<stdexcept>
#ifdef DEBUG
#include<iostream>
#endif

#include<headers/culinalg-vector.hpp>
#include<headers/culinalg-exceptions.hpp>

#include<sources/culinalg-cuheader.cuh>

void clg::Vector::alloc_irepr_throw_()
{
    // Set irepr_ to valid temp state
    irepr_->reset();
    
    // Try to allocate. We keep the local variables void here, unnecessary to do implicit conversion
    void *_h_data, *_d_data;
    clg::wrapCudaError<clg::HostAllocationFailedException>(cudaMallocHost(&_h_data, dim_*sizeof(float)));
    clg::wrapCudaError<clg::DeviceAllocationFailedException>(cudaMalloc(&_d_data, dim_*sizeof(float)));

    // Allocation passed
    irepr_->host_data = _h_data;
    irepr_->device_data = _d_data;
    
#ifdef DEBUG
    std::cout << "Allocated host " << _h_data << " device " << _d_data << std::endl;
#endif
}

void clg::Vector::delloc_irepr_throw_()
{
    // Ensure valid state
    void* _h_data = irepr_->host_data;
    void* _d_data = irepr_->device_data;
    irepr_->reset();

    // Attempt to free, throw if fail
    if(_h_data) clg::wrapCudaError<clg::HostDellocationFailedException>(cudaFreeHost(_h_data));
    if(_d_data) clg::wrapCudaError<clg::DeviceDellocationFailedException>(cudaFree(_d_data));

#ifdef DEBUG
    std::cout << "Dellocated host " << _h_data << " device " << _d_data << std::endl;
#endif
}

clg::Vector::Vector(size_t n, bool init) : dim_(n)
{
#ifdef DEBUG
    std::cout << "Creating a new vector of size " << n << std::endl;
#endif
    // Make new CuData
    irepr_ = new CuData();

    // Try allocating
    alloc_irepr_throw_();

    // Set to 0 on the CPU
    float* _h_floats = (float*)irepr_->host_data;
    if(init) for(size_t i = 0; i < dim_; ++i) _h_floats[i] = 0.0;
}

clg::Vector::~Vector()
{
#ifdef DEBUG
    std::cout << "Deleting vector at host " << irepr_->host_data << " device " <<
        irepr_->device_data << std::endl;
#endif   
    // Free data
    if(irepr_->host_data)
        cudaFreeHost(irepr_->host_data);
    if(irepr_->device_data)
        cudaFree(irepr_->device_data);

    // delte irepr_
    delete irepr_;
}

clg::Vector::Vector(const clg::Vector& other) : dim_(other.dim_)
{
 #ifdef DEBUG
    std::cout << "Copy constructing new vector " << std::endl;
#endif
   // Make new CuData
    irepr_ = new CuData();
    
    // Try to allocate data
    alloc_irepr_throw_();

    // Copy data
    copyCuData(*irepr_, *(other.irepr_), dim_*sizeof(float));
}

clg::Vector::Vector(clg::Vector&& other) : dim_(other.dim_)
{
#ifdef DEBUG
    std::cout << "Move constructing new vector " << std::endl;
#endif

    // Make new CuData
    irepr_ = new CuData();

    // Just move the data pointers in irepr_
    irepr_->move_from(*(other.irepr_));
    
    // Leave other in valid state, not pointing to same data.
    irepr_->reset();
}

clg::Vector& clg::Vector::operator=(const clg::Vector& other)
{
 #ifdef DEBUG
    std::cout << "Copy assigning new vector " << std::endl;
#endif
   // Check dimensionality
    if(dim_ != other.dim_) throw clg::DimensionalityMismatchException(dim_, other.dim_);

    // Attempt to delete data in this, maintain strong exception guarantee
    delloc_irepr_throw_();

    // Copy data
    copyCuData(*irepr_, *(other.irepr_), dim_*sizeof(float));
    
    // Return this
    return *this;
}

clg::Vector& clg::Vector::operator=(clg::Vector&& other)
{
 #ifdef DEBUG
    std::cout << "Move assigniing vector " << std::endl;
#endif
   // Check dimensionality
    if(dim_ != other.dim_) throw clg::DimensionalityMismatchException(dim_, other.dim_);

    // Attempt to delete data in this, maintain strong exception guarantee
    delloc_irepr_throw_();

    // Just move the data pointers in irepr_
    irepr_->move_from(*(other.irepr_));
    
    // Reset other.irepr_ to prevent double referencing
    other.irepr_->reset();

    // Return this
    return *this;
}


float& clg::Vector::operator[](size_t index)
{
 #ifdef DEBUG
    //std::cout << "Accessing Vector " << std::endl;
#endif
   // Check bounds
    if(index >= dim_) throw std::out_of_range("Out of range access on vector");

    // Synchronize memory
    irepr_->memsync_host(dim_ * sizeof(float));

    return ((float*)irepr_->host_data)[index];
}


/**
 * Kernel to interpret the passed float* x, y as device addresses for two vectors of dimension dim
 * represented as contagious arrays
 * and stores the result in r, which is assumed to be a pointer to a memory with enough space for
 * the resulting vector.
 */
__global__ void kern_vec_add_(float* x, float* y, float* r, size_t dim)
{
    size_t _strd = blockDim.x * gridDim.x;
    for(size_t _i = blockIdx.x * blockDim.x + threadIdx.x; _i < dim; _i += _strd)
        r[_i] = x[_i] + y[_i];
#ifdef DEBUG
    //printf("add kern\n");
#endif
}

clg::Vector clg::operator+(const clg::Vector& x, const clg::Vector& y)
{
#ifdef DEBUG
    std::cout << "Adding vectors " << std::endl;
#endif

    // Check dimensionality
    if(x.dim_ != y.dim_) throw clg::DimensionalityMismatchException(x.dim_, y.dim_);

    // Synchronize both

    x.irepr_->memsync_device(x.dim_ * sizeof(float));
    y.irepr_->memsync_device(y.dim_ * sizeof(float));
    
    // Create clg::Vector to store result
    clg::Vector _vec(x.dim_, false);

    // Execute kernel, grid size is chosen so that there are as many blocks as multiprocessors.
    int _grid_size;
    int _device;
    cudaGetDevice(&_device);
    cudaDeviceGetAttribute(&_grid_size, cudaDevAttrMultiProcessorCount, _device);
    kern_vec_add_<<<_grid_size, CULINALG_BLOCK_SIZE>>>(
        (float*)x.irepr_->device_data, (float*)y.irepr_->device_data,
        (float*)_vec.irepr_->device_data, x.dim_);
    clg::wrapCudaError<clg::KernelLaunchFailedException>(cudaGetLastError());
    cudaDeviceSynchronize();
    clg::wrapCudaError<clg::KernelSynchronizationFailedException>(cudaGetLastError());
    
    // Device memory is correct
    _vec.irepr_->host_data_synced = false;
    
    // Return
    return _vec;
}

clg::Vector& clg::Vector::operator+=(const clg::Vector& other)
{
    *this = *this + other;
    return *this;
}
