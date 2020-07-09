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

To use this library, it is necessary to build against it for the target system of user using nvidia's 
cuda compiler. To do so, write standard C++ code but save it as cuda files with a `.cu` extension
into some directory we will call `<source-files>`, then on unix like systems run:

```
nvcc -I <culinalg-directory> -o <output-binary-name> <culinalg-directory>/sources/*.cu <source-files>
```

On windows, a similar command may be used on the console, or an IDE may be set up to work
equivalently.

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
* Cache variables should be `CamelCase`.
* Interface variables should be `BLOCK_CASE`.
