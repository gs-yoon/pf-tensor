#include "t32f.h"
#include "t32f_generic.h"

bool pf_t32f_init(pf_tensor* self)
{
    self->type   = PF_FLOAT32;

    self->mul    = pf_mul32f;
    self->add    = pf_add32f;
    self->sub    = pf_sub32f;
    self->div    = pf_div32f;
    self->dot    = pf_dot32f;
    self->matMul = pf_matMul32f;

    self->at = pf_t32f_at; // TODO : change to vargs
}

double pf_t32f_at(pf_tensor * data, int dim, ...)
{
    int pos[10] = {0};
    VA_IDX(dim, pos)

    switch(dim)
    {
        case 0:  return ATD0(float,data->root);
        case 1:  return ATDN(float,data->root, data->shape, pos, 1);
        case 2:  return ATDN(float,data->root, data->shape, pos, 2);
        case 3:  return ATDN(float,data->root, data->shape, pos, 3);
        case 4:  return ATDN(float,data->root, data->shape, pos, 4);
        case 5:  return ATDN(float,data->root, data->shape, pos, 5);
        case 6:  return ATDN(float,data->root, data->shape, pos, 6);
        case 7:  return ATDN(float,data->root, data->shape, pos, 7);
        case 8:  return ATDN(float,data->root, data->shape, pos, 8);
        case 9:  return ATDN(float,data->root, data->shape, pos, 9);
    }

    // int idx = pos[dim-1];
    // int shape_size = 1;
    // for (int i = dim-1  ; i > 0 ; i--)
    // {
    //     shape_size *= data->shape[i];
    //     idx += shape_size * pos[i-1];
    // }
    // return (double)(*(float32*)data->root + idx);

}
/*
*/

int pf_t32f_aligned_alloc(pf_tensor* self)
{
    self->root = (void*)aligned_alloc((size_t)16*sizeof(char), (size_t)self->size*sizeof( float ));

    return (self->root != NULL) ?  1 : 0;
}