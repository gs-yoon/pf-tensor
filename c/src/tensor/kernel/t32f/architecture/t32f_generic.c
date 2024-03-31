#include "t32f_generic.h"

bool pf_mul32f(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    if(result->root == NULL)
    {
        pf_t32f_init(result, self->device);
        pf_t32f_alloc(result,  self->ndim, self->shape);
    }
    else
    {
        PF_LOG("reulst is not empty tensor");
        return 0;
    }

    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result->root;

    for (int i =0 ; i < self->size ; i++)
        c[i] = a[i] * b[i];

    return 1;
}

bool pf_add32f(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    if(result->root == NULL)
    {
        pf_t32f_init(result, self->device);
        pf_t32f_alloc(result,  self->ndim, self->shape);
    }
    else
    {
        PF_LOG("reulst is not empty tensor");
        return 0;
    }

    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result->root;

    for (int i =0 ; i < self->size ; i++)
        c[i] = a[i] + b[i];

    return 1;
}
bool pf_sub32f(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    if(result->root == NULL)
    {
        pf_t32f_init(result, self->device);
        pf_t32f_alloc(result,  self->ndim, self->shape);
    }
    else
    {
        PF_LOG("reulst is not empty tensor");
        return 0;
    }

    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result->root;

    for (int i =0 ; i < self->size ; i++)
        c[i] = a[i] - b[i];

    return 1;
}
bool pf_div32f(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    if(result->root == NULL)
    {
        pf_t32f_init(result, self->device);
        pf_t32f_alloc(result,  self->ndim, self->shape);
    }
    else
    {
        PF_LOG("reulst is not empty tensor");
        return 0;
    }

    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result->root;

    for (int i =0 ; i < self->size ; i++)
        c[i] = a[i] / b[i];

    return 1;
}
bool pf_dot32f(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    int nshape[10] = {0};
    memcpy(nshape, self->shape, self->size);
    nshape[self->ndim-1] = 1;

    if(result->root == NULL)
    {
        pf_t32f_init(result, self->device);
        pf_t32f_alloc(result,  self->ndim, self->shape);
    }
    else
    {
        PF_LOG("reulst is not empty tensor");
        return 0;
    }

    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result->root;

    return 1;
}
bool pf_matMul32f(pf_tensor* self, pf_tensor *operand, pf_tensor* result)
{}
