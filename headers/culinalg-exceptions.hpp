/**
 * Exception classes for culinalg
 */

#ifndef CULINALG_HEADER_EXCEPTIONS
#define CULINALG_HEADER_EXCEPTIONS

#include <stdexcept>

namespace clg
{
    /**
     * This exception occurs when memory could not be allocated, iether on the host side or on the
     * device side. Extended to define seperate exceptions for host and device memory allocation
     * errors.
     */
    class AllocationFailedException : std::runtime_error
    {
        explicit AllocationFailedException (const char* str) : std::runtime_error(str) {}
        explicit AllocationFailedException (const std::string& str) : std::runtime_error(str) {}
    };
    /**
     * A class representing an AllocationFailedException occuring when trying to allocate host
     * memory
     */
    class HostAllocationFailedException : AllocationFailedException
    {
        explicit HostAllocationFailedException (const char* str) : 
            AllocationFailedException(str) {}
        explicit HostAllocationFailedException (const std::string& str) : 
            AllocationFailedException(str) {}
    };
    /**
     * A class representing an AllocationFailedException occuring when trying to allocate device
     * memory
     */
    class DeviceAllocationFailedException : AllocationFailedException
    {
        explicit DeviceAllocationFailedException (const char* str) : 
            AllocationFailedException(str) {}
        explicit DeviceAllocationFailedException (const std::string& str) : 
            AllocationFailedException(str) {}
    };

    // SEE ALSO: template<T> void wrapCudaError(CudaError_t) in 
}

#endif
