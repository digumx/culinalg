#ifndef CULINALG_HEADER_CORE
#define CULINALG_HEADER_CORE

namespace clg
{
    /**
     * A collection of flags representing the status of the current vector or matrix
     */
    struct ObjectStatus
    {
        bool host_memory_sync;              // Host memory holds correct value
        bool device_memory_sync;            // Device memory holds correct value
        bool under_read_compute;            // Being read as a part of an on-going computation
        bool under_write_compute;           // Being written as part of on-going computation
    }
}

#endif
