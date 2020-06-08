/*
 * Implements vector
 */

#include<headers/culinalg-vector>

#include<sources/culinalg-object>

clg::Vector::Vector(size_t n) : dim_(n)
{
    // Make new CuObject
    irepr_ = new CuObject();

    // Set irepr_ to valid temp stat
    irepr_->host_data = NULL;
    irepr_->device_data = NULL;
}
