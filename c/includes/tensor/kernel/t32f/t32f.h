#ifndef __T32F_H__
#define __T32F_H__


#include "pf_body.h"

bool pf_t32f_init(pf_tensor* self, PF_DEVICE device);
bool pf_t32f_set(pf_tensor* self, double value);
void pf_t32f_to(pf_tensor* self, PF_DEVICE device);
int  pf_t32f_alloc(pf_tensor* self, int dim, int* shape);
pf_tensor pf_t32f_at(pf_tensor* self, int dim, ...);

int pf_t32f_malloc(pf_tensor* self);
int pf_t32f_aligned_alloc(pf_tensor* self);
double pf_t32f_values(pf_tensor* self);

bool pf_t32f_mul(pf_tensor* self, pf_tensor* operand, pf_tensor* result);
bool pf_t32f_add(pf_tensor* self, pf_tensor* operand, pf_tensor* result);
bool pf_t32f_sub(pf_tensor* self, pf_tensor* operand, pf_tensor* result);
bool pf_t32f_div(pf_tensor* self, pf_tensor* operand, pf_tensor* result);
bool pf_t32f_dot(pf_tensor* self, pf_tensor* operand, pf_tensor* result);
bool pf_t32f_matmul(pf_tensor* self, pf_tensor* operand, pf_tensor* result);

#endif