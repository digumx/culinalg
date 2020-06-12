/**
 * Class and methods for representing a large dimensional vector.
 */

#ifndef CULINALG_HEADER_VECTOR
#define CULINALG_HEADER_VECTOR

namespace clg
{
    /*
     * Forward declaration of CuObject
     */
    template<class T> struct CuObject;

    // TODO add double precision support
    /**
     * A very large dimensional vector with operators overloaded for vector addition.
     */
    class Vector                                    
    {
        public:
            /**
             * Construct a 0 vector of dimensionality n.
             */
            Vector(size_t n);
            /**
             * Destruct a vector
             */
            ~Vector();
            /**
             * Copy and move constructors
             */
            Vector(const Vector& other);
            Vector(Vector&& other);
            /**
             * Copy and move assignment operators
             */
            Vector operator=(const Vector& other);
            Vector operator=(Vector&& other);
            /**
             * Friend function to copy vectors
             */
            friend void copyVector(const Vector& dst, const Vector& src);
            /**
             * Friend function for vector addition
             */
            friend Vector operator+(const Vector& x, const Vector& y);
            /**
             * Compound operator for vector addition
             */
            Vector operator+=(const Vector& other);
        
        private:
            CuObject<float>* irepr_;
            size_t dim_;
    };

    /**
     * Adds two vectors together. Friend to Vector.
     * @param x The left argument
     * @param y The right argument
     */
    Vector operator+(const Vector& x, const Vector& y);

    /**
     * Copies vector data from second argument into first.
     */
    void copyVector(const Vector& dst, const Vector& src);
}

#endif
