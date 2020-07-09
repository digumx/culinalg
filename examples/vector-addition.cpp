#include <headers/culinalg.hpp>

#include <iostream>

int main(int argc, char** argv)
{
    clg::Vector v1(10);
    clg::Vector v2(10);
    for(int i = 0; i < 10; i++)
    {
        v1[i] = 1;
        v2[i] = 2;
    }

    clg::Vector v3(10);
    //for(int i = 0; i < 10; i++)
    //    v3[i] = 5;

    v3 = v1 + v2;

    for(int i = 0; i < 10; i++)
        std::cout << v3[i] << std::endl;
}
