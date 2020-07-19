# culinalg

A CUDA linear algebra library written in C++. This is mainly a personal exploration of some parallel
algorithms for common linear algebraic operations.

## Usage

The following code snippet demonstrates usage: 
``` 
#include <headers/culinalg.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    clg::Vector v1(1000);                   // Define two vectors of size 1000
    clg::Vector v2(1000);
    for(int i = 0; i < 10; i++)             // Set value of the vectors
    {
        v1[i] = 1;
        v2[i] = 2;
    }

    clg::Vector v3(1000);                   // Define a new vector of same size as v1 and v2
    v3 = v1 + v2;                           // Add v1 and v2 and set it to v3

    bool correct = true;
    for(int i = 0; i < 10; i++)
        correct &= v3[i] == 3;              // Check if result is correct
    std::cout << 
        (correct ? "Correct" : "Wrong") << 
        std::endl; 
}
```

## Building

The recommended way to build this library is by bundling it into the user's project via CMake, as
the API is still unstable. A standalone build is provided, however the build does not perform any
system wide installation. For build configuration options, see build options below.

The buildsystem was tested on Linux for generating unix makefiles, however, it should theoretically
work on any platform supported by CMake.

### Standalone

To build the library in a standalone manner, simply configure with cmake and run make. Clone the
repository, say to `<culinalg-root>`, create the root folder of the build tree, and on Linux simply
do:

```
cmake ..
make
```

Under the default configuration settings this will build culinalg and all examples provided.

To use the generated `libculinalg.so`, if the `Culinalg_Static_Link_Cudart` option is on (which is
the default), one can simply link to it. Thus, to compile the vector addition example from a build
tree rooted at `<culinalg-root>/build` one would do:

```
g++ -I ../ -L./ -o vector-addition ../examples/vector-addition.cpp -lculinalg
```

### Bundled using CMake

This library can be bundled into any CMake project. This is the recommended way to use it. To bundle
this library, clone the repository to a subdirectory of your project, say at `<project-root>/extlib/culinalg`
and add it to your `CMakeLists.txt` as a subdirectory. Configuration options may be set before the
`add_subdirectory` command. Then the variables `CULINALG_SOURCE_DIRS` and `CULINALG_LIBRARIES` will
become available which may be used in the usual way. The following configures `culinalg` to
dynamically link to the cuda runtime and links it to `Target`:

```
set(Culinalg_Static_Link_Cudart OFF)
add_subdirectory(extlib/culinalg)

target_include_directories(Target PRIVATE ${CULINALG_INCLUDE_DIRS})
target_link_libraries(Target PRIVATE ${CULINALG_LIBRARIES})
```

Note that when bundling into a CMake project, the examples are never built.

### Build Options

The following build options are configurable via CMake:

* `Culinalg_Gpu_Binary_Architectures` : A semicolon seperated list of real architecture numbers for
  which GPU binary code should be generated. The compiled binary will essenctially be a fatbinary
  containing binary GPU code for all of these architectures.
* `Culinalg_Gpu_Ptx_Architecure` : The virtual architecture number for which to generate ptx code.
  This allows for forward compatibility of the generated binary with future GPU architectures.
  This should be a very high number, but all lower virual architectures should be covered as real
  architectures in the previous option. In both this and the previous option the numbers for the
  architectures should correspond to the codes used by Nvidia, see the gpu and virtual architecture
  lists in the nvcc docs.
* `Culinalg_Static_Link_Cudart` : Should the cuda runtime library be statically linked into the
  produced binary.
* `Culinalg_Build_Examples` : Should the examples be built. Ignored if being bundled into another
  CMake projects, in which case no examples are built.

## Styling

The following code styling conventions are being followed for c, c++ and cuda code:

* Namespaces should be `lowercasewithnowordseperation`, and short if possible.
* Classes, structs and types should be `CamelCaseWithNoWordSeperation`.
* Macros should be `BLOCK_CASE_WITH_UNDERSCORE_WORD_SEPERATION`.
* Global and static variables should be `Spine_Case_With_Capitilization_Of_First_Letter`.
* Non member functions should be `camelCaseWithSmallStartingLetter`.
* Public members and function arguements should be `spine_case`.
* Private members and filescope references should be `spine_case_ending_with_underscore_`.
* Local variables should be `_spine_case_begining_with_underscore`.

For CMake, the styling is as follows:

* Non cache variables should be `spine_case`.
* Cache variables should be `Camel_Spine_Case`.
* Project names, target names etc should use `SpineCase`
* Interface variables should be `BLOCK_CASE`.
