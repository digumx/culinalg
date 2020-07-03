/**
 * Implements several functions declared in source/culinalg-cuheader.cuh
 */

#ifdef DEBUG
#include <iostream>
#endif

#include<sources/culinalg-cuheader.cuh>
#include<headers/culinalg-exceptions.hpp>

void clg::copyCuData(const CuData& dst, const CuData& src, size_t count)
{
    // Check that CuDatas point to valid data.
    if(!dst.host_data || !src.host_data || !dst.device_data || !src.host_data)
        throw clg::CopyFailedException("CuData is invalid or points to no data");

    // Perform copy
    cudaError_t err; 
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
    host_data = src.host_data;
    device_data = src.device_data;
    host_data_synced = src.host_data_synced;
}

void clg::CuData::memsync_host(size_t size)
{
    // Early return
    if(host_data_synced) return;

#ifdef DEBUG
    std::cout << "Syncing to host " << host_data << " from device " << device_data << " size " <<
        size << std::endl;
#endif

    // Try copying. If copy fails, treating source as correct seems safe
    clg::wrapCudaError<clg::CopyFailedException>(cudaMemcpy(host_data, device_data, size,
                cudaMemcpyDeviceToHost));

    // Set sync flags
    host_data_synced = true;
}

void clg::CuData::memsync_device(size_t size)
{
    // Early return
    if(!host_data_synced) return;

#ifdef DEBUG
    std::cout << "Syncing from host " << host_data << " to device " << device_data << " size " <<
        size << std::endl;
#endif

    // Try copyin. If copy fails, treating source as correct seems safeg
    clg::wrapCudaError<clg::CopyFailedException>(cudaMemcpy(device_data, host_data, size,
                cudaMemcpyHostToDevice));

    // Set sync flags
    host_data_synced = false;
}
