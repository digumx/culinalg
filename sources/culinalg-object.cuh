#ifndef CULINALG_HEADER_OBJECT
#define CULINALG_HEADER_OBJECT

#include<queue>

namespace clg
{
    /**
     * A class representing a large CUDA object with memory beinged mirrored between host and device.
     * Used to store the data for vectors and matrices. Template parameter T is expected to be iether
     * float or double.
     */
    template<class T>
    struct CuObject
    {
        /*
         * Pointers to data for host and device. Should iether be null, or refer to valid memory
         */
        T* host_data; 
        T* device_data;
        /*
         * Events reading to or writing from this object. Both cannot be nonempty, as reading to and
         * writing from the same object simultaneously should not happen
         */
        std::queue<CudaEvent_t> reader_events;
        CudaEvent_t writer_events;
    };
}

#endif
