#include "t32f.h"
#include "t32f_generic.h"

pf_operator pf_t32f_mul_in_device[] = {pf_mul32f_generic};
pf_operator pf_t32f_add_in_device[] = {pf_add32f_generic};
pf_operator pf_t32f_sub_in_device[] = {pf_sub32f_generic};
pf_operator pf_t32f_div_in_device[] = {pf_div32f_generic};
pf_operator pf_t32f_dot_in_device[] = {pf_dot32f_generic};
pf_operator pf_t32f_matmul_in_device[] = {pf_matmul32f_generic};

bool pf_t32f_init(pf_tensor* self, PF_DEVICE device)
{
    self->root   = NULL;
    self->shape  = NULL;
    self->size   = 0;
    self->type   = PF_FLOAT32;
    self->device = device;

    self->mul    = pf_t32f_mul;
    self->add    = pf_t32f_add;
    self->sub    = pf_t32f_sub;
    self->div    = pf_t32f_div;
    self->dot    = pf_t32f_dot;
    self->matmul = pf_t32f_matmul;

    self->at     = pf_t32f_at;
    self->set    = pf_t32f_set;
    self->values = pf_t32f_values;
}

double pf_t32f_values(pf_tensor* self)
{
    return (double)*(float32*)self->root;
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
        for(int i =0 ; i<self->size ; i++)
            *(float32*)(ndata.root+i) = *(float32*)(self->root+i);

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
    result.size = 1; 

    switch(dim)
    {
        case 0:  result.root = (void*)ATD0(float,self->root); break;
        case 1:  result.root = (void*)ATDN(float,self->root, self->shape, pos, 1); break;
        case 2:  result.root = (void*)ATDN(float,self->root, self->shape, pos, 2); break;
        case 3:  result.root = (void*)ATDN(float,self->root, self->shape, pos, 3); break;
        case 4:  result.root = (void*)ATDN(float,self->root, self->shape, pos, 4); break;
        case 5:  result.root = (void*)ATDN(float,self->root, self->shape, pos, 5); break;
        case 6:  result.root = (void*)ATDN(float,self->root, self->shape, pos, 6); break;
        case 7:  result.root = (void*)ATDN(float,self->root, self->shape, pos, 7); break;
        case 8:  result.root = (void*)ATDN(float,self->root, self->shape, pos, 8); break;
        case 9:  result.root = (void*)ATDN(float,self->root, self->shape, pos, 9); break;
        default :
            PF_LOG("dimension error");
    }

    return result;
}

bool pf_t32f_set(pf_tensor* self, double value)
{
    float32* data = (float32*)self->root;
    printf("%d\n",self->size);
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

/* operator */
bool pf_t32f_mul(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    if(self->type != PF_FLOAT32 || operand->type != PF_FLOAT32)
        return 0;
    if(self->device != operand->device)
        return 0;

    pf_t32f_mul_in_device[self->device](self,operand,result);
}

bool pf_t32f_add(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    if(self->type != PF_FLOAT32 || operand->type != PF_FLOAT32)
        return 0;
    if(self->device != operand->device)
        return 0;

    pf_t32f_add_in_device[self->device](self,operand,result);
}

bool pf_t32f_sub(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    if(self->type != PF_FLOAT32 || operand->type != PF_FLOAT32)
        return 0 ;
    if(self->device != operand->device)
        return 0;

    pf_t32f_sub_in_device[self->device](self,operand,result);
    return 1; 
}

bool pf_t32f_div(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    if(self->type != PF_FLOAT32 || operand->type != PF_FLOAT32)
        return 0;
    if(self->device != operand->device)
        return 0;

    pf_t32f_div_in_device[self->device](self,operand,result);
    return 1; 
}

bool pf_t32f_dot(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    if(self->type != PF_FLOAT32 || operand->type != PF_FLOAT32)
        return 0;
    if(self->device != operand->device)
        return 0;

    pf_t32f_dot_in_device[self->device](self,operand,result);
    return 1; 
}

bool pf_t32f_matmul(pf_tensor* self, pf_tensor* operand, pf_tensor* result)
{
    if(self->type != PF_FLOAT32 || operand->type != PF_FLOAT32)
        return 0;
    if(self->device != operand->device)
        return 0;

    pf_t32f_matmul_in_device[self->device](self,operand,result);
    return 1; 
}