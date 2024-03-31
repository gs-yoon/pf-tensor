#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "t32f_generic.h"

/***************************
 * MACRO Examples (in nargs.h)
   - AT(float,tensor, A,B,C)
   - makeTensor(SHAPE(A,B,C))
*****************************/

void initTensor(pf_tensor* self, PF_TYPE type, PF_DEVICE device);
bool allocTensor(pf_tensor* self, int dim, int* shape);

pf_tensor makeTensor(PF_TYPE type, int dim, ...);
pf_tensor makeZeros(PF_TYPE type, int dim, ...);
bool breakTensor(pf_tensor* self);
//bool makeOnes(pf_tensor* tensor, PF_TYPE type, int dim, ...);
bool breakTensor(struct pf_tensor* tensor);
bool freeTensor(pf_tensor* self);

#endif

