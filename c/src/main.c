#include "tensor.h"
#include "pf_utils.h"
#include "stdio.h"


int main()
{
    pf_tensor tensor;
    tensor = makeZeros(PF_FLOAT32, SHAPE(2,3));
    pf_tensor b = makeZeros(PF_FLOAT32, SHAPE(2,3));
    
    tensor.set(&tensor, 1);
    b.set(&b,2);

    // pf_tensor result = (tensor.add(&tensor, &b));
    pf_tensor result = pf_sub(&tensor, &b);
    printf("test\n");
    pfprint(&tensor);
    pfprint(&result);
    if( freeTensor(&tensor) == false)
        printf("free failed\n");
    else
        printf("free success\n");
    return 0;
}