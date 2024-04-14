#include "t32f.h"
#include "t32f_generic.h"

bool pf_t32f_init(pf_tensor* self, PF_DEVICE device)
{
    self->root   = NULL;
    self->shape  = NULL;
    self->size   = 0;
    self->type   = PF_FLOAT32;
    self->device = device;

    if (device == PF_GENERIC)
    {
        self->mul    = pf_mul32f;
        self->add    = pf_add32f;
        self->sub    = pf_sub32f;
        self->div    = pf_div32f;
        self->dot    = pf_dot32f;
        self->matmul = pf_matmul32f;
    }

    self->at  = pf_t32f_at; // TODO : change to vargs
    self->set = pf_t32f_set;
}

void pf_t32f_to(pf_tensor* self, PF_DEVICE device)
{

    pf_t32f_init(self, device);
    if (self->root != NULL)
    {
        pf_tensor ndata;
        pf_t32f_init(&ndata, device);
        pf_t32f_alloc(&ndata, self->ndim, self->shape);
        
        //copy self to ndata
            //for ~
        free(self->root);
        free(self->shape);
        *self = ndata;
    }
    return;
}

pf_tensor pf_t32f_at(pf_tensor * self, int dim, ...)
{
    int pos[10] = {0};
    VA_IDX(dim, pos)

    pf_tensor result;
    static int constant_shape[1] = {1};
    pf_t32f_init(&result, self->device );
    result.shape = constant_shape;
    result.ndim = 1;

    switch(dim)
    {
        case 0:  result.root = ATD0(float,self->root);
        case 1:  result.root = ATDN(float,self->root, self->shape, pos, 1);
        case 2:  result.root = ATDN(float,self->root, self->shape, pos, 2);
        case 3:  result.root = ATDN(float,self->root, self->shape, pos, 3);
        case 4:  result.root = ATDN(float,self->root, self->shape, pos, 4);
        case 5:  result.root = ATDN(float,self->root, self->shape, pos, 5);
        case 6:  result.root = ATDN(float,self->root, self->shape, pos, 6);
        case 7:  result.root = ATDN(float,self->root, self->shape, pos, 7);
        case 8:  result.root = ATDN(float,self->root, self->shape, pos, 8);
        case 9:  result.root = ATDN(float,self->root, self->shape, pos, 9);
    }
    return result;
}

bool pf_t32f_set(pf_tensor* self, double value)
{
    float32* data = (float32*)self->root;
    for (int i =0 ; i < self->size; i ++)
        data[i] = (float32)value;
}

int pf_t32f_alloc(pf_tensor* self,  int dim, int* shape)
{
    if(self->root != NULL)
    {
        PF_LOG("not empty tensor. already allocated");
        return 0;
    }

    self->ndim   = dim;
    self->shape  = (int*)malloc(sizeof(int) * dim);
    memcpy(self->shape, shape, sizeof(int)*dim);
    
    self->size =1;
    for(int i =0 ; i < dim ; i++)
        self->size *= shape[i];

    int ret = 0;
    if(self->device == PF_GENERIC)
        ret = pf_t32f_malloc(self);
    else if(self->device == PF_X86)
        ret = pf_t32f_aligned_alloc(self);
    else
        PF_LOG("Unkown Device");
    return ret;
}

int pf_t32f_malloc(pf_tensor* self)
{
    self->root = (void*)malloc((size_t)self->size*sizeof( float32 ));
    return (self->root != NULL) ?  1 : 0;
}

int pf_t32f_aligned_alloc(pf_tensor* self)
{
    self->root = (void*)aligned_alloc((size_t)16*sizeof(char), (size_t)self->size*sizeof( float32 ));

    return (self->root != NULL) ?  1 : 0;
}