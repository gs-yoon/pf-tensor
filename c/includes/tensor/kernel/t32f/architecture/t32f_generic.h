#ifndef __T32F_GENERIC_H__
#define __T32F_GENERIC_H__

#include "t32f.h"

bool pf_mul32f_generic(pf_tensor* self,       pf_tensor* operand, pf_tensor* result );
bool pf_add32f_generic(pf_tensor* self,       pf_tensor* operand, pf_tensor* result );
bool pf_sub32f_generic(pf_tensor* self,       pf_tensor* operand, pf_tensor* result );
bool pf_div32f_generic(pf_tensor* self,       pf_tensor* operand, pf_tensor* result );
bool pf_dot32f_generic(pf_tensor* self,       pf_tensor* operand, pf_tensor* result );
bool pf_matmul32f_generic(pf_tensor* self,    pf_tensor* operand, pf_tensor* result );

#endif