/**
 * Exception classes for culinalg
 */

#ifndef CULINALG_HEADER_EXCEPTIONS
#define CULINALG_HEADER_EXCEPTIONS

#include <stdexcept>
#include <string>

namespace clg
{
    /**
     * This exception occurs when memory could not be allocated, iether on the host side or on the
     * device side. Extended to define seperate exceptions for host and device memory allocation
     * errors.
     */
    class AllocationFailedException : std::runtime_error
    {
        explicit AllocationFailedException (const char* str) : 
            std::runtime_error(("Allocation Failed: " + std::string(str)).c_str()) {}
        explicit AllocationFailedException (const std::string& str) : 
            std::runtime_error("Allocation Failed:" + str) {}
    };
    /**
     * A class representing an AllocationFailedException occuring when trying to allocate host
     * memory
     */
    class HostAllocationFailedException : AllocationFailedException
    {
        explicit HostAllocationFailedException (const char* str) : 
            AllocationFailedException(("Host Allocation Failed: " + std::string(str)).c_str()) {}
        explicit HostAllocationFailedException (const std::string& str) : 
            AllocationFailedException("Host Allocation Failed:" + str) {}
    };
    /**
     * A class representing an AllocationFailedException occuring when trying to allocate device
     * memory
     */
    class DeviceAllocationFailedException : AllocationFailedException
    {
        explicit DeviceAllocationFailedException (const char* str) : 
            AllocationFailedException(("Device Allocation Failed: " + std::string(str)).c_str()) {}
        explicit DeviceAllocationFailedException (const std::string& str) : 
            AllocationFailedException("Device Allocation Failed:" + str) {}
    };

    // SEE ALSO: wrapCudaError() in culinalg-cuheader.cuh
}

#endif
