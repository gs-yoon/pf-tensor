#include "tensor.h"


void initTensor(pf_tensor* self, PF_TYPE type, PF_DEVICE device)
{
    if (type == PF_FLOAT32)
        pf_t32f_init(self, device); // TODO : add device parameter
}

bool allocTensor(pf_tensor* self, int dim, int* shape)
{
    if (self->type == PF_FLOAT32)
        return pf_t32f_alloc(self, dim, shape);
    else
        return 0;
}

static inline void createTensor(pf_tensor* tensor, PF_TYPE type, PF_DEVICE device, int dim, int* shape )
{
    initTensor(tensor, type, device);
    int success = allocTensor(tensor, dim, shape);
    if(!success)
        PF_LOG("Allocation Failed");
}

static inline void createTensorLike(pf_tensor* result, pf_tensor* wannabe)
{
    initTensor(result, wannabe->type, wannabe->device);
    int success = allocTensor(result, wannabe->ndim, wannabe->shape);
    if(!success)
        PF_LOG("Allocation Failed");
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
bool breakTensor(pf_tensor* self)
{
    freeTensor(self);
}

pf_tensor makeTensor(PF_TYPE type, int dim, ...)
{
    pf_tensor tensor;
    int shape[10] = {0};
    VA_IDX(dim,shape);

    createTensor(&tensor, type, PF_GENERIC, dim, shape);

    return tensor;
}

pf_tensor makeTensorIn(PF_TYPE type, PF_DEVICE device ,int dim, ...)
{
    pf_tensor tensor;
    int shape[10] = {0};
    VA_IDX(dim,shape);

    createTensor(&tensor, type, PF_GENERIC, dim, shape);

    return tensor;
}


pf_tensor makeZeros( PF_TYPE type, int dim, ...)
{
    pf_tensor tensor;
    int shape[10] = {0};
    VA_IDX(dim,shape);

    createTensor(&tensor, type, PF_GENERIC, dim, shape);
    
    if (type == PF_FLOAT32)
        memset(tensor.root, 0x00, (size_t)tensor.size * sizeof(float32));
    
    return tensor;
}

pf_tensor makeZerosIn(PF_TYPE type, PF_DEVICE device ,int dim, ...)
{
    pf_tensor tensor;
    int shape[10] = {0};
    VA_IDX(dim,shape);

    createTensor(&tensor, type, PF_GENERIC, dim, shape);

    if (type == PF_FLOAT32)
        memset(tensor.root, 0x00, (size_t)tensor.size * sizeof(float32));

    return tensor;
}


pf_tensor pf_add(pf_tensor* self, pf_tensor* operand)
{
    pf_tensor result;
    createTensorLike(&result, self);

    self->add(self,operand,&result);
    return result;
}
pf_tensor pf_mul(pf_tensor* self, pf_tensor* operand)
{
    pf_tensor result;
    createTensorLike(&result, self);

    self->mul(self,operand,&result);
    return result;
}
pf_tensor pf_sub(pf_tensor* self, pf_tensor* operand)
{
    pf_tensor result;
    createTensorLike(&result, self);

    self->sub(self,operand,&result);
    return result;
}
pf_tensor pf_div(pf_tensor* self, pf_tensor* operand)
{
    pf_tensor result;
    createTensorLike(&result, self);

    self->div(self,operand,&result);
    return result;
}
pf_tensor pf_dot(pf_tensor* self, pf_tensor* operand)
{
    pf_tensor result;
    createTensorLike(&result, self);

    self->dot(self,operand,&result);
    return result;
}
pf_tensor pf_matmul(pf_tensor* self, pf_tensor* operand)
{
    pf_tensor result;
    createTensorLike(&result, self);

    self->matMul(self,operand,&result);
    return result;
}
