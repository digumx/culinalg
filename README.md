# culinalg

A CUDA linear algebra library written in C++. This is mainly a personal exploration of some parallel
algorithms for common linear algebraic operations.

## Usage

The following code snippet demonstrates usage:

```
clg::Vector v1(1000);                       // Define three 0 vectors of dimensionality 1000
clg::Vector v2(1000);
clg::Vector v3(1000);

for(int i = 0; i < 1000; ++i)               // Initialize vectors
{
    v1[i] = 1;
    v2[i] = 2;
}

v3 = v1 + v2;                               // Just add them

for(int i = 0; i < 1000; ++i)               // print out sum
    std::cout << v3[i] << std::endl;
```


## Building

The recommended way to build this library is by bundling it into the user's project via CMake, as
the API is still unstable. A standalone build is provided, however the build is not configured to
perform any system wide installation. For build configuration options, see build options below.

### Standalone

To build the library in a standalone manner, simply configure with cmake and run make. Clone the
repository, say to `<culinalg-root>`, create the root folder of the build tree, and on Linux simply
do:

```
cmake ..
make
```

Under the default configuration settings this will build culinalg and all examples provided.

To use the 

### Bundled

This library can be bundled into any CMake project.

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
