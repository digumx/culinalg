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
    class AllocationFailedException : public std::runtime_error
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
    class HostAllocationFailedException : public AllocationFailedException
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
    class DeviceAllocationFailedException : public AllocationFailedException
    {
        explicit DeviceAllocationFailedException (const char* str) : 
            AllocationFailedException(("Device Allocation Failed: " + std::string(str)).c_str()) {}
        explicit DeviceAllocationFailedException (const std::string& str) : 
            AllocationFailedException("Device Allocation Failed:" + str) {}
    };
    // SEE ALSO: wrapCudaError() in culinalg-cuheader.cuh

    /**
     * A class representing an error that occurs when dimensionality of operands in a binary
     * operation do not properly match. This is thrown, for example, when adding two vectors of
     * different dimensionalities or multiplying matrices where the number of columns of the left
     * matrix does not match the number of rows of the right matrix.
     */
    class DimensionalityMismatchException : public std::logic_error
    {
        /**
         * The following constructor is used to create a new DimensionalityMismatchException to
         * report mismatch in dimensions when working with two vectors.
         */
        DimensionalityMismatchException(int dim_a, int dim_b) :
            logic_error("Dimensionality Mismatch: Attempting to operate on vectors of dimensions " +
                    std::to_string(dim_a) + " and " + std::to_string(dim_b)) {}
    }
}

#endif
