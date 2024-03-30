#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "t32f_generic.h"

/***************************
 * MACRO Examples (in nargs.h)
   - AT(float,tensor, A,B,C)
   - makeTensor(SHAPE(A,B,C))
*****************************/
void initTensor(pf_tensor* self, PF_TYPE type);
pf_tensor* makeTensor(PF_TYPE type, int ndim, ...); 
pf_tensor* makeZeros(PF_TYPE type, int ndim, ...);
pf_tensor* makeOnes(PF_TYPE type, int ndim, ...);
bool breakTensor(struct pf_tensor* tensor);

#endif

