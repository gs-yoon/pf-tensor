#ifndef __SHAPE_H__
#define __SHAPE_H__
#include <initializer_list>
#include "type_definition.h"
#include <algorithm>
#include <iostream>
class Shape
{
    /* data */
    private:
        int* data_  = nullptr;
        int  size_  = 0;
        int  ndim_  = 0;
    public:
        int ndim();
        int operator[](int idx);
        int size();
        void clear();

        Shape();
        ~Shape();
        Shape(std::initializer_list<int32> s);
};

Shape::Shape()
{
}
Shape::~Shape()
{
    //clear();
}
Shape::Shape(std::initializer_list<int32> s)
{
    ndim_ = s.size();
    data_ = (int*)malloc( sizeof(int32) * ndim_);
    std::copy(s.begin(), s.end(), data_);
}

inline int Shape::ndim()
{
    return ndim_;
}
inline int Shape::operator[](int idx)
{
    return data_[idx];
}
inline int Shape::size()
{
    if (size_ == 0)
    {
        int size = 1;
        for (int i = 0; i < ndim_; i++)
                size *= data_[i];
        size_ = size;
    }
    return size_;
}
inline void Shape::clear()
{
    if (data_ != nullptr)
        free(data_);
    data_ = nullptr;
    size_ = 0;
    ndim_ = 0;
}




#endif