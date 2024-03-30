#ifndef __T32F_H__
#define __T32F_H__


#include "pf_body.h"

bool pf_t32f_init(pf_tensor* self);
double pf_t32f_at(pf_tensor* self, int* pos);
int pf_t32f_aligned_alloc(pf_tensor* self);

#endif