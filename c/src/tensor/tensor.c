#include "tensor.h"


void initTensor(pf_tensor* self, PF_TYPE type)
{
    if (type == PF_FLOAT32)
        pf_t32f_init(self);
}

bool allocTensor(pf_tensor* self, int dim, int* shape)
{
    self->ndim = dim;
    self->shape = (int*)malloc(sizeof(int) * dim);
    memcpy(self->shape, shape, sizeof(int)*dim);
    
    self->size =1;
    for(int i =0 ; i < dim ; i++)
        self->size *= shape[i];

    if (self->type == PF_FLOAT32)
        return pf_t32f_aligned_alloc(self);
    else
        return 0;
}

bool freeTensor(pf_tensor* self)
{
    int ret =0;
    if (self-> shape != NULL)
    {
        free(self->shape);
    }
    if (self-> root != NULL)
    {
        free(self->root);
        ret =1;
    }
    return ret;
}

bool makeTensor(pf_tensor* self, PF_TYPE type, int dim, ...)
{
    int shape[10] = {0};
    VA_IDX(dim,shape);

    initTensor(self, type);
    return allocTensor(self, dim, shape);;
}

bool breakTensor(pf_tensor* self)
{
    freeTensor(self);
}


bool makeZeros(pf_tensor* self, PF_TYPE type, int dim, ...)
{
    int shape[10] = {0};
    VA_IDX(dim,shape);

    initTensor(self, type);
    int ret = allocTensor(self, dim, shape);

    if (ret == 0 )
        return ret;
    
    if (type == PF_FLOAT32)
        memset(self->root, 0x00, (size_t)self->size * sizeof(float32));
    
    return ret;
}

void pfprint(pf_tensor* self)
{
    pfprintShape(self);
    printf("size = %d \n", self->size);
    // for (int i =0 ; i < self->shape[0] ; i++) 
    printf("%d \n", (int)AT(float32,self,1,1)); // TODO : for type
}
void pfprintShape(pf_tensor* self)
{
    printf("shape = (");
    for (int i =0 ; i < self->ndim ; i++)
        printf("%d,", self->shape[i]);
    printf(")\n");
}
