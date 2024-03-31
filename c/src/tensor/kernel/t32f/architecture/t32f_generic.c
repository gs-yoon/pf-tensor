#include "t32f_generic.h"

pf_tensor pf_mul32f(pf_tensor* self, pf_tensor* operand)
{
    pf_tensor result;
    pf_t32f_init(&result, self->device);
    pf_t32f_alloc(&result,  self->ndim, self->shape);

    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result.root;

    for (int i =0 ; i < self->size ; i++)
        c[i] = a[i] * b[i];

    return result;
}

pf_tensor pf_add32f(pf_tensor* self, pf_tensor* operand)
{
    pf_tensor result;
    pf_t32f_init(&result, self->device);
    pf_t32f_alloc(&result,  self->ndim, self->shape);

    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result.root;

    for (int i =0 ; i < self->size ; i++)
        c[i] = a[i] + b[i];

    return result;
}
pf_tensor pf_sub32f(pf_tensor* self, pf_tensor* operand)
{
    pf_tensor result;
    pf_t32f_init(&result, self->device);
    pf_t32f_alloc(&result,  self->ndim, self->shape);

    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result.root;

    for (int i =0 ; i < self->size ; i++)
        c[i] = a[i] - b[i];

    return result;
}
pf_tensor pf_div32f(pf_tensor* self, pf_tensor* operand)
{
    pf_tensor result;
    pf_t32f_init(&result, self->device);
    pf_t32f_alloc(&result,  self->ndim, self->shape);

    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result.root;

    for (int i =0 ; i < self->size ; i++)
        c[i] = a[i] / b[i];

    return result;
}
pf_tensor pf_dot32f(pf_tensor* self, pf_tensor* operand)
{
    int nshape[10] = {0};
    memcpy(nshape, self->shape, self->size);
    nshape[self->ndim-1] = 1;

    pf_tensor result;
    pf_t32f_init(&result, self->device);
    pf_t32f_alloc(&result,  self->ndim, self->shape);

    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result.root;

    return result;
}
pf_tensor pf_matMul32f(pf_tensor* self, pf_tensor *operand)
{}
