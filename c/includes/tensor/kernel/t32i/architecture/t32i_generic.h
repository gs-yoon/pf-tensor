#ifndef __T32I_GENERIC_H__
#define __T32I_GENERIC_H__

#include "t32i.h"

pf_tensor* pf_mul32i(pf_tensor* self,       pf_tensor* operand);
pf_tensor* pf_add32i(pf_tensor* self,       pf_tensor* operand);
pf_tensor* pf_sub32i(pf_tensor* self,       pf_tensor* operand);
pf_tensor* pf_div32i(pf_tensor* self,       pf_tensor* operand);
pf_tensor* pf_dot32i(pf_tensor* self,       pf_tensor* operand);
pf_tensor* pf_matMul32i(pf_tensor* self,    pf_tensor *operand);

#endif