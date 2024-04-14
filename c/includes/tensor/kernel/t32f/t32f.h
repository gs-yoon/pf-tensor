#ifndef __T32F_H__
#define __T32F_H__


#include "pf_body.h"

bool pf_t32f_init(pf_tensor* self, PF_DEVICE device);
pf_tensor pf_t32f_at(pf_tensor* self, int dim, ...);
bool pf_t32f_set(pf_tensor* self, double value);
void pf_t32f_to(pf_tensor* self, PF_DEVICE device);
int pf_t32f_alloc(pf_tensor* self, int dim, int* shape);

int pf_t32f_malloc(pf_tensor* self);
int pf_t32f_aligned_alloc(pf_tensor* self);
#endif