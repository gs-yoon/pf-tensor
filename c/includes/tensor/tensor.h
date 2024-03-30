#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "t32f_generic.h"

/***************************
 * MACRO Examples (in nargs.h)
   - AT(float,tensor, A,B,C)
   - makeTensor(SHAPE(A,B,C))
*****************************/

void initTensor(pf_tensor* self, PF_TYPE type);
bool allocTensor(pf_tensor* self, int dim, int* shape);

bool makeTensor(pf_tensor* tensor, PF_TYPE type, int dim, ...);
bool breakTensor(pf_tensor* self);
bool makeZeros(pf_tensor* tensor, PF_TYPE type, int dim, ...);
//bool makeOnes(pf_tensor* tensor, PF_TYPE type, int dim, ...);
bool breakTensor(struct pf_tensor* tensor);
bool freeTensor(pf_tensor* self);


void pfprint(pf_tensor* self);
void pfprintShape(pf_tensor* self);

#endif

