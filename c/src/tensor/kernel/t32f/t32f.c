#include "t32f.h"
#include "t32f_generic.h"

bool pf_t32f_init(pf_tensor* self)
{
    self->mul    = pf_mul32f;
    self->add    = pf_add32f;
    self->sub    = pf_sub32f;
    self->div    = pf_div32f;
    self->dot    = pf_dot32f;
    self->matMul = pf_matMul32f;

    self->at = pf_t32f_at; // TODO : change to vargs
}

double pf_t32f_at(pf_tensor * data, int* pos)
{
    switch(data->info.ndim)
    {
        case 0:  return ATD0(float,data->root);
        case 1:  return ATDN(float,data->root, data->info.shape, pos, 1);
        case 2:  return ATDN(float,data->root, data->info.shape, pos, 2);
        case 3:  return ATDN(float,data->root, data->info.shape, pos, 3);
        case 4:  return ATDN(float,data->root, data->info.shape, pos, 4);
        case 5:  return ATDN(float,data->root, data->info.shape, pos, 5);
        case 6:  return ATDN(float,data->root, data->info.shape, pos, 6);
        case 7:  return ATDN(float,data->root, data->info.shape, pos, 7);
        case 8:  return ATDN(float,data->root, data->info.shape, pos, 8);
        case 9:  return ATDN(float,data->root, data->info.shape, pos, 9);
        default: // TODO : change to using va_args 
        {
            // int dim_1 = data->info.ndim-1;
            // int idx = pos[dim_1];

            // int shape_size = 1;
            // for (int i = dim_1  ; i > 0 ; i--)
            // {
                // shape_size *= data->info.shape[i];
                // idx += shape_size * pos[i-1];
            // }
            // return AT(double,data->root, data->info.shape, pos, 4);
        }
    }
}

int pf_t32f_aligned_alloc(pf_tensor* self)
{
    self->root = (void*)aligned_alloc((size_t)16*sizeof(char), (size_t)self->info.size*sizeof( float ));
}