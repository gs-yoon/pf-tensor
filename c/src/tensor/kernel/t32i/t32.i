#include "t32i.h"
#include "t32i_generic.h"

bool pf_t32i_init(pf_tensor* self)
{
    self->type   = PF_FLOAT32;

    self->mul    = pf_mul32i;
    self->add    = pf_add32i;
    self->sub    = pf_sub32i;
    self->div    = pf_div32i;
    self->dot    = pf_dot32i;
    self->matmul = pf_matmul32i;

    self->at = pf_t32i_at;
}

double pf_t32i_at(pf_tensor * data, int dim, ...)
{
    int pos[10] = {0};
    VA_IDX(dim, pos)

    switch(dim)
    {
        case 0:  return ATD0(int32,data->root);
        case 1:  return ATDN(int32,data->root, data->shape, pos, 1);
        case 2:  return ATDN(int32,data->root, data->shape, pos, 2);
        case 3:  return ATDN(int32,data->root, data->shape, pos, 3);
        case 4:  return ATDN(int32,data->root, data->shape, pos, 4);
        case 5:  return ATDN(int32,data->root, data->shape, pos, 5);
        case 6:  return ATDN(int32,data->root, data->shape, pos, 6);
        case 7:  return ATDN(int32,data->root, data->shape, pos, 7);
        case 8:  return ATDN(int32,data->root, data->shape, pos, 8);
        case 9:  return ATDN(int32,data->root, data->shape, pos, 9);
    }
}

int pf_t32i_aligned_alloc(pf_tensor* self)
{
    self->root = (void*)aligned_alloc((size_t)16*sizeof(char), (size_t)self->size*sizeof( int32 ));

    return (self->root != NULL) ?  1 : 0;
}