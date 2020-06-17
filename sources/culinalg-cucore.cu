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

void clg::copyCuData(const CuData& dst, const CuData& src, size_t count)
{
    // Check that CuDatas point to valid data.
    if(!dst.host_data || !src.host_data || !dst.device_data || !src.host_data)
        throw clg::CopyFailedException("CuData is invalid or points to no data");

    // Perform copy
    CudaError_t err; 
    if(dst.host_data_synced)
    {
        if(src.host_data_synced)
            err = cudaMemcpy(dst.host_data, src.host_data, count, cudaMemcpyHostToHost); //TODO bench
        else
            err = cudaMemcpy(dst.host_data, src.device_data, count, cudaMemcpyDeviceToHost);
    }
    else
    {
        if(src.host_data_synced)
            err = cudaMemcpy(dst.device_data, src.host_data, count, cudaMemcpyHostToDevice);        
        else    //TODO benc the following
            err = cudaMemcpy(dst.device_data, src.device_data, count, cudaMemcpyDeviceToDevice);
    }

    // Check for error
    clg::wrapCudaError<clg::CopyFailedException>(err); 
}

void clg::CuData::reset()
{
    host_data = nullptr;
    device_data = nullptr;
    host_data_synced = true;
}

void clg::CuData::move_from(const CuData& src)
{
    host_data = other.host_data;
    device_data = other.device_data;
    host_data_synced = other.host_data_synced;
}

void clg::CuData::memsync_host(size_t size)
{
    // Early return
    if(host_data_synced) return;

    // Try copying
    clg::wrapCudaError<clg::CopyFailedException>(cudaMemcpy(host_data, device_data, size,
                cudaMemcpyDeviceToHost));

    // Set sync flags
    host_data_synced = true;
}

void clg::CuData::memsync_device(size_t size)
{
    // Early return
    if(!host_data_synced) return;

    // Try copying
    clg::wrapCudaError<clg::CopyFailedException>(cudaMemcpy(device_data, host_data, size,
                cudaMemcpyHostToDevice));

    // Set sync flags
    host_data_synced = false;
}
