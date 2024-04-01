#ifndef __T32F_H__
#define __T32F_H__


#include "pf_body.h"

pf_tensor* pf_t32i_init(pf_tensor* self, PF_TYPE type);
double pf_t32i_at(pf_tensor* self, ...);
int pf_t32i_alloc(pf_tensor* self);

#endif