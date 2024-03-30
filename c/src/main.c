#include "tensor.h"
#include "stdio.h"



int main()
{
    pf_tensor tensor;
    makeZeros(&tensor,PF_FLOAT32, SHAPE(2,3));
    printf("test\n");
    pfprint(&tensor);
    if( freeTensor(&tensor) == false)
        printf("free failed\n");
    else
        printf("free success\n");
    return 0;
}