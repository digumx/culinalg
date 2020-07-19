# 
# HEADER 
#

cmake_minimum_required( VERSION 3.12 )

project( CulinalgProject LANGUAGES CXX CUDA )




# 
# CONFIGURABLE SETTINGS
#

# Define the build type as a cache entry
if( NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE "Release" CACHE STRING "The type of build. May be Debug or Release" FORCE )
    set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" )
endif()

# Generate binaries from ptx of the same version for each of these architectures. The architecture
# numbers should point to both a valid virtual and a valid real architecture.
if( NOT Culinalg_Gpu_Binary_Architectures )
    set( Culinalg_Gpu_Binary_Architectures 35 50 53 60 61 62 70 72 
        CACHE STRING "Semicolon seperated list of gpu real architectures for which to generate gpu \
                      binary code"
        )
endif()

# Generate ptx but no binary codes for this architecture. This is so that the compiled binary is
# guaranteed to be compatible with future gpu generations
if( NOT Culinalg_Gpu_Ptx_Architecture )
    set( Culinalg_Gpu_Ptx_Architecture 75 CACHE STRING "Architecture for which to generate gpu ptx code"
        )
endif()

# Link to cuda runtime statically or dynamically
if( NOT Culinalg_Static_Link_Cudart )
    option( Culinalg_Static_Link_Cudart "Turn on to link to the cuda runtime statically" ON )
endif()

# Build examples or not ignored if not a standalone build, examples are not built
if( NOT Culinalg_Build_Examples )
    option( Culinalg_Build_Examples "Turn on to build examples." ON )
endif()




#
# CULINALG FILE LIStS
#

# List of source files for culinalg
set( culinalg_sources   ${CMAKE_CURRENT_SOURCE_DIR}/sources/culinalg-cucore.cu
                        ${CMAKE_CURRENT_SOURCE_DIR}/sources/culinalg-vector.cu )

# List of private headers for culinalg
set( culinalg_header_deps       ${CMAKE_CURRENT_SOURCE_DIR}/sources/culinalg-cuheader.cuh 
                                ${CMAKE_CURRENT_SOURCE_DIR}/headers/culinalg-exceptions.hpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/headers/culinalg-vector.hpp 
                                ${CMAKE_CURRENT_SOURCE_DIR}/headers/culinalg.hpp )
                              



# 
# GPU ARCHITECTURE SUPPORT AND CODE GENRATION
#

# Generate string representing -gencode commands to pass to nvcc
message( "Configuring gencode options to pass to cuda compiler. Assuming nvcc compatible cli." )
set( gencode_lines "" )
foreach( arch IN LISTS Culinalg_Gpu_Binary_Architectures )
    set( gencode_lines "${gencode_lines}-gencode arch=compute_${arch},code=sm_${arch} " )
endforeach( arch )
set( gencode_lines "${gencode_lines}-gencode \
        arch=compute_${Culinalg_Gpu_Ptx_Architecture},code=compute_${Culinalg_Gpu_Ptx_Architecture} " )
message( "Using gencode options ${gencode_lines}" )




# 
# CULINALG BUILD CONFIG
#

# Build configurations
set( CMAKE_CONFIGURATION_TYPES "Release" "Debug" )

# Culinalg library targets
add_library( CulinalgTarget SHARED  ${culinalg_sources} 
                                    ${culinalg_header_deps} )

# Names for library targets
set_target_properties( CulinalgTarget PROPERTIES OUTPUT_NAME culinalg )

# Defines
target_compile_definitions( CulinalgTarget
        PRIVATE "$<$<CONFIG:Debug>:DEBUG>" )

# Include directoreis for library targets
target_include_directories( CulinalgTarget PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )

# Common properties to both targets
set_target_properties( CulinalgTarget PROPERTIES
                        CXX_STANDARD 11
                        CUDA_SEPARABLE_COMPILATION ON )

# Linking CUDA runtime
if( Culinalg_Static_Link_Cudart )
    set_target_properties( CulinalgTarget PROPERTIES CUDA_RUNTIME_LIBRARY Static )
else()
    set_target_properties( CulinalgTarget PROPERTIES CUDA_RUNTIME_LIBRARY Shared )
endif()

# Set gencode lines
target_compile_options( CulinalgTarget PRIVATE 
                        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:${gencode_lines}>>" )



#
# EXPORT VARIABLES
#

if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )

    # Culinalg include dirs
    set( CULINALG_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} )

    # Libraries
    set( CULINALG_LIBRARIES CulinalgTarget )

else()

    # Culinalg include dirs
    set( CULINALG_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE )
    # Libraries
    set( CULINALG_LIBRARIES CulinalgTarget PARENT_SCOPE )

endif()
message( "Library link arguements: ${CULINALG_LIBRARIES}" )
message( "Include directoreis: ${CULINALG_INCLUDE_DIRS}" )


#
# EXAMPLES
#

if( Culinalg_Build_Examples AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR ) 

    # Vector addition
    add_executable( VectorAddition examples/vector-addition.cpp )
    set_target_properties( VectorAddition PROPERTIES OUTPUT_NAME vector-addition )
    target_include_directories( VectorAddition PRIVATE ${CULINALG_INCLUDE_DIRS} )
    target_link_libraries( VectorAddition ${CULINALG_LIBRARIES} )

endif()
