#include <headers/culinalg.hpp>
#include <iostream>

#define N 1<<20

int main(int argc, char** argv)
{
    clg::Vector v1(N);                                          // Define two vectors of size N
    clg::Vector v2(N);
    for(int i = 0; i < N; i++)                                  // Fill them
    {
        v1[i] = 1;
        v2[i] = 2;
    }

    clg::Vector v3(N);                                          // Define another vector of size N

    v3 = v1 + v2;                                               // Add and store the vectors

    bool correct = true;
    for(int i = 0; i < N; i++)                                  // Check if vectors are correct
        correct &= v3[i] == 3;
    std::cout << (correct ? "OK" : "Wrong") << std::endl;
}
