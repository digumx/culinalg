# 
# HEADER 
#

cmake_minimum_required( VERSION 3.12 )

project( Culinalg_Project LANGUAGES CXX CUDA )




# 
# CONFIGURABLE SETTINGS
#

# Generate binaries from ptx of the same version for each of these architectures. The architecture
# numbers should point to both a valid virtual and a valid real architecture.
set( CulinalgCodeGenerationArchitectures 35 50 53 60 61 62 70 72 75 
    CACHE STRING "Semicolon seperated list of gpu real architectures for which to generate gpu code"
    )

# Link to cuda runtime statically or dynamically
option( CulinalgStaticLinkCudart "Turn on to link to the cuda runtime statically" OFF ) 




#
# CULINALG FILE LIStS
#

# List of source files for culinalg
set( culinalg_sources   ${CMAKE_CURRENT_SOURCE_DIR}/sources/culinalg-cucore.cu
                        ${CMAKE_CURRENT_SOURCE_DIR}/sources/culinalg-vector.cu )

# List of private headers for culinalg
set( culinalg_private_headers   ${CMAKE_CURRENT_SOURCE_DIR}/sources/culinalg-cuheader.cuh 
                                ${CMAKE_CURRENT_SOURCE_DIR}/headers/culinalg-exceptions.hpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/headers/culinalg-vector.hpp )
# List of interface headers for culinalg 
set( CULINALG_HEADERS   ${CMAKE_CURRENT_SOURCE_DIR}/headers/culinalg.hpp ) 




# 
# GPU ARCHITECTURE SUPPORT AND CODE GENRATION
#

# Generate string representing -gencode commands to pass to nvcc
message(Configuring gencode options to pass to cuda compiler. Assuming nvcc compatible cli.)
set( gencode_lines "" )
foreach( arch IN LISTS CulinalgCodeGenerationArchitectures )
    set( gencode_lines "${gencode_lines}-gencode arch=compute_${arch},code=sm_${arch} " )
endforeach( arch )
message( Using gencode options ${gencode_lines} )




# 
# CULINALG BUILD CONFIG
#

# Culinalg library targets
add_library( Culinalg_Static STATIC ${culinalg_sources} 
                                    ${culinalg_private_headers}
                                    ${CULINALG_HEADERS} )
add_library( Culinalg_Shared SHARED ${culinalg_sources} 
                                    ${culinalg_private_headers}
                                    ${CULINALG_HEADERS} )

# Names for library targets
set_target_properties( Culinalg_Static PROPERTIES OUTPUT_NAME culinalg_static )
set_target_properties( Culinalg_Shared PROPERTIES OUTPUT_NAME culinalg )

# Include directoreis for library targets
target_include_directories( Culinalg_Static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )
target_include_directories( Culinalg_Shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )

# Common properties to both targets
set_target_properties( Culinalg_Static Culinalg_Shared PROPERTIES
                        CXX_STANDARD 11
                        CUDA_SEPARABLE_COMPILATION ON )

# Linking CUDA runtime
if( CulinalgStaticLinkCudart )
    set_target_properties( Culinalg_Shared Culinalg_Static PROPERTIES CUDA_RUNTIME_LIBRARY Shared )
else()
    set_target_properties( Culinalg_Shared Culinalg_Static PROPERTIES CUDA_RUNTIME_LIBRARY Shared )
endif()

# Set gencode lines
target_compile_options( Culinalg_Static PRIVATE
                        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:${gencode_lines}>>" )



#
# TESTS AND EXAMPLES
#

# Test1
add_executable( Test1 tests/test1.cu )
set_target_properties( Test1 PROPERTIES OUTPUT_NAME test1 )
target_link_libraries( Test1 PRIVATE Culinalg_Shared )
