/**
 * A common header containing forward declarations and type definitions dependent on the cuda
 * runtime
 */

#ifndef CULINALG_HEADER_CUHEADER
#define CULINALG_HEADER_CUHEADER

#include<queue>
#include<string>

namespace clg
{
    /**
     * A class representing a large CUDA object with memory beinged mirrored between host and device.
     * Used to store the data for vectors and matrices. Template parameter T is expected to be iether
     * float or double.
     */
    struct CuObject
    {
        /**
         * Pointers to data for host and device. Should iether be nullptr, or refer to valid memory
         */ 
        void* host_data; 
        void* device_data;
        /**
         * The following flag is true if the host_data contains the correct data for the vector,
         * otherwise the device_data contains the correct data.
         */
        bool host_data_synced;

        // TODO Use for async
        ///**
        // * Events reading to or writing from this object. Both cannot be nonempty, as reading to and
        // * writing from the same object simultaneously should not happen
        // */
        //std::queue<CudaEvent_t> reader_events;
        //CudaEvent_t writer_events;
    };

    /**
     * A function that wraps a CUDA call to check for errors and if one occurs it throws the error
     * of the passed template arguement type. Template arguement is expected to be a valid error
     * defined in headers/culinalg-exceptions.hpp. Note that this function itself does not provide
     * any exception guarantee, and is intended to simply be syntactic sugar for the often repeated
     * if-throw pattern for wrapping CudaErrors in exceptions.
     */
    template<class E> inline void wrapCudaError(const CudaError_t& err);

    /**
     * Copies data from the second CuObject arguement into the first.
     */
    void copyCuObject(const CuObject& dst, const CuObject& src);
    


    /**
     * A thin wrapper class for CudaEvent_t.
     */
    // TODO complete
    //class CuEvent
    //{
    //    public:
    //        /**
    //         * Constructs a new cuda event, calls cudaEventCreate()
    //         */
    //        CuEvent() { cudaEventCreate(&event); }
    //        /**
    //         * Destroys an event, calls 

    //    private:
    //        CudaEvent_t event;
    //};
}

#endif
