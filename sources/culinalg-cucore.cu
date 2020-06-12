/**
 * Implements several functions declared in source/culinalg-cuheader.cuh
 */

#include<sources/culinalg-cuheader.cuh>

template<class E> inline void clg::wrapCudaError(const CudaError_t& err)
{
    if(err != cudaSuccess)
        throw E("CUDA Error: " + std::string(cudaGetErrorName(err)) + ": " +
                std::string(cudaGetErrorString(err)));
}

void clg::copyCuObject(const CuObject& dst, const CuObject& src)
{
    CudaError_t err; 
    if(dst.irepr_->host_data_synced)
    {
        if(src.irepr_->host_data_synced)
            err = cudaMemcpy(dst.irepr_->host_data, src.irepr_->host_data, cudaMemcpyHostToHost); //TODO bench
        else
            err = cudaMemcpy(dst.irepr_->host_data, src.irepr_->device_data, cudaMemcpyDeviceToHost);
    }
    else
    {
        if(src.irepr_->host_data_synced)
            err = cudaMemcpy(dst.irepr_->device_data, src.irepr_->host_data, cudaMemcpyHostToDevice);        
        else
            err = cudaMemcpy(dst.irepr_->device_data, src.irepr_->device_data, cudaMemcpyDeviceToDevice);
    }

    // Check for error
    clg::wrapCudaError<clg::CopyFailedException>(err); 
}
