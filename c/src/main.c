#include "tensor.h"
#include "pf_utils.h"
#include "stdio.h"


int main()
{
    pf_tensor tensor;
    tensor = makeZeros(PF_FLOAT32, SHAPE(2,2));
    pf_tensor b = makeZeros(PF_FLOAT32, SHAPE(2,2));
    
    tensor.set(&tensor, 1);
    b.set(&b,1);
    
    int ee[2][1]= {0};

    // pf_tensor result = (tensor.add(&tensor, &b));
    pf_tensor result = pf_matmul(&tensor, &b);
    printf("test\n");
    pfprint(tensor);
    pfprint(result);
    // pf_tensor t = result.at(&result,SHAPE(1,1));
    result.at(&result,S(1,1)).set(&result,1);
    pfprint(result);
    // pfprint( (result.at(&result,SHAPE(1,1))) );
    if( freeTensor(&tensor) == false)
        printf("free failed\n");
    else
        printf("free success\n");
    return 0;
}