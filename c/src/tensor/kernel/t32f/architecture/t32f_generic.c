#include "t32f_generic.h"

bool pf_mul32f_generic(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result->root;

    for (int i =0 ; i < self->size ; i++)
        c[i] = a[i] * b[i];
}

bool pf_add32f_generic(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result->root;

    for (int i =0 ; i < self->size ; i++)
        c[i] = a[i] + b[i];
}
bool pf_sub32f_generic(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result->root;

    for (int i =0 ; i < self->size ; i++)
        c[i] = a[i] - b[i];
}
bool pf_div32f_generic(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result->root;

    for (int i =0 ; i < self->size ; i++)
        c[i] = a[i] / b[i];
}
bool pf_dot32f_generic(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    int bucket_size = self->shape[self->ndim-1];
    
    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result->root;

    int o = -1;
    for (int i =0 ; i < self->size ; i++)
    {
        if (i % bucket_size == 0 )
            o++;
        c[o] += a[i] * b[i];
    }
}
bool pf_matmul32f_generic(pf_tensor* self, pf_tensor *operand, pf_tensor* result)
{
    float32* a = (float32*)self->root;
    float32* b = (float32*)operand->root;
    float32* c = (float32*)result->root;
    if( self->ndim == 2 )
    {
        for (int k =0; k < self->shape[1]; k ++)
        {
            for (int i =0 ; i< self->shape[0] ; i++)
            {
                float32 tmp = ATS(a,self->shape,i,k);
                for (int j =0 ; j < operand->shape[1]; j++)
                {
                    ATS(c,result->shape,i,j) += ATS(b,operand->shape,k,j) * tmp;
                }
            }
        }
    }
}
