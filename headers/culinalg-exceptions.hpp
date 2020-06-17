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
     * This exception occurs when a cuda memcpy fails.
     */
    class CopyFailedException : public std::runtime_error
    {
        explicit CopyFailedException (const char* str) : 
            std::runtime_error(("Copy Failed: " + std::string(str)).c_str()) {}
        explicit CopyFailedException (const std::string& str) : 
            std::runtime_error("Copy Failed:" + str) {}
    };
    /**
     * This exception occurs when memory could not be dellocated, iether on the host side or on the
     * device side. Extended to define seperate exceptions for host and device memory allocation
     * errors.
     */
    class DellocationFailedException : public std::runtime_error
    {
        explicit DellocationFailedException (const char* str) : 
            std::runtime_error(("Dellocation Failed: " + std::string(str)).c_str()) {}
        explicit DellocationFailedException (const std::string& str) : 
            std::runtime_error("Dellocation Failed:" + str) {}
    };
    /**
     * A class representing an DellocationFailedException occuring when trying to allocate host
     * memory
     */
    class HostDellocationFailedException : public DellocationFailedException
    {
        explicit HostDellocationFailedException (const char* str) : 
            DellocationFailedException(("Host Dellocation Failed: " + std::string(str)).c_str()) {}
        explicit HostDellocationFailedException (const std::string& str) : 
            DellocationFailedException("Host Dellocation Failed:" + str) {}
    };
    /**
     * A class representing an DellocationFailedException occuring when trying to allocate device
     * memory
     */
    class DeviceDellocationFailedException : public DellocationFailedException
    {
        explicit DeviceDellocationFailedException (const char* str) : 
            DellocationFailedException(("Device Dellocation Failed: " + std::string(str)).c_str()) {}
        explicit DeviceDellocationFailedException (const std::string& str) : 
            DellocationFailedException("Device Dellocation Failed:" + str) {}
    };

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
        /*
         * The following constructor is used to create an exception reporting a dimensionality
         * mismatch between two vectors in the what() message
         */
        DimensionalityMismatchException(int dim_a, int dim_b) :
            logic_error("Dimensionality Mismatch: Attempting to operate on vectors of dimensions " +
                    std::to_string(dim_a) + " and " + std::to_string(dim_b)) {}
    }
}

#endif
