# 
# HEADER 
#

cmake_minimum_required( VERSION 3.12 )

project( CulinalgProject LANGUAGES CXX CUDA )




# 
# CONFIGURABLE SETTINGS
#

# Generate binaries from ptx of the same version for each of these architectures. The architecture
# numbers should point to both a valid virtual and a valid real architecture.
set( Culinalg_GPU_Code_Generation_Architectures 35 50 53 60 61 62 70 72 75 
    CACHE STRING "Semicolon seperated list of gpu real architectures for which to generate gpu code"
    )

# Link to cuda runtime statically or dynamically
option( Culinalg_Static_Link_Cudart "Turn on to link to the cuda runtime statically" ON ) 

# Build culinalg dynamically
option( Culinalg_Build_Dynamic "Turn on to build culinalg into a dynamically linked library" ON )

# Build examples or not ignored if not a standalone build, examples are not built
option( Culinalg_Build_Examples "Turn on to build examples." ON )




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
foreach( arch IN LISTS Culinalg_GPU_Code_Generation_Architectures )
    set( gencode_lines "${gencode_lines}-gencode arch=compute_${arch},code=sm_${arch} " )
endforeach( arch )
message( "Using gencode options ${gencode_lines}" )




# 
# CULINALG BUILD CONFIG
#

# Culinalg library targets
if( Culinalg_Build_Dynamic )
    add_library( CulinalgTarget SHARED  ${culinalg_sources} 
                                        ${culinalg_header_deps} )
else()
    add_library( CulinalgTarget STATIC  ${culinalg_sources} 
                                        ${culinalg_header_deps} )
endif()

# Names for library targets
set_target_properties( CulinalgTarget PROPERTIES OUTPUT_NAME culinalg )

# Include directoreis for library targets
target_include_directories( CulinalgTarget PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )

# Common properties to both targets
set_target_properties( CulinalgTarget PROPERTIES
                        CXX_STANDARD 11
                        CUDA_SEPARABLE_COMPILATION ON )

# Linking CUDA runtime
if( Culinalg_Static_Link_Cudart )
    set_target_properties( CulinalgTarget PROPERTIES CUDA_RUNTIME_LIBRARY Shared )
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
