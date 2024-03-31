#ifndef __T32F_GENERIC_H__
#define __T32F_GENERIC_H__

#include "t32f.h"

pf_tensor pf_mul32f(pf_tensor* self,       pf_tensor* operand);
pf_tensor pf_add32f(pf_tensor* self,       pf_tensor* operand);
pf_tensor pf_sub32f(pf_tensor* self,       pf_tensor* operand);
pf_tensor pf_div32f(pf_tensor* self,       pf_tensor* operand);
pf_tensor pf_dot32f(pf_tensor* self,       pf_tensor* operand);
pf_tensor pf_matMul32f(pf_tensor* self,    pf_tensor *operand);

#endif