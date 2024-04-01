#ifndef __TENSOR_Hc__
#define __TENSOR_Hc__

#include "type_definition.h"
#include "shape.h"
#include "tensor.h"
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


template < typename T = float32> 
class Tensor
{
private:
    /* data */
    T* root_ = nullptr;
    Shape shape_;

public:
    Tensor(/* args */);
    Tensor(Shape shape);
    ~Tensor();

    bool makeZeros(Shape shape);
    bool makeTensor(Shape shape);
    void breakTensor();
    int getndim();
    int getShape();
    Tensor<T> matmul();
    Tensor<T> dotmul();
    Tensor<T> operator*(Tensor<T>);
    Tensor<T> operator+(Tensor<T>);
    Tensor<T> operator-(Tensor<T>);
    Tensor<T> operator/(Tensor<T>);
    Tensor<T> operator()(...);
    // T operator[](...);
    void print();
};


template <typename T> 
Tensor<T>::Tensor(/* args */)
{}

template <typename T> 
Tensor<T>::~Tensor()
{
    breakTensor();
}

//template<typename T>
template <typename T> 
void Tensor<T>::breakTensor()
{
    if (root_ != nullptr)
    {
        free(root_);
        root_ = nullptr;
    }

    //shape_.clear();
}

template <typename T> 
Tensor<T>::Tensor(Shape shape)
{
    makeTensor(shape);
}

template <typename T>
bool Tensor<T>::makeZeros(Shape shape)
{
    makeTensor(shape);
    memset(root_, 0x00, (size_t)shape.size() * sizeof(T));
}

template <typename T>
bool Tensor<T>::makeTensor(Shape shape)
{
    printf("d\n");
    root_ = (T*)aligned_alloc((size_t)16*sizeof(char), (size_t)shape.size()*sizeof(T));
    shape_ = shape;
}
template <typename T>
void Tensor<T>::print()
{
    printf("size = %d \n", shape_.size());
    for (int i =0 ; i < shape_.size() ; i++)
        printf("%f " , root_[i]);
}

#endif
