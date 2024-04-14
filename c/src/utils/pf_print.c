
#include "pf_print.h"

void pfprint(pf_tensor self)
{
    pfprintShape(self);

    int rank = self.ndim;
    int* shape = self.shape;
    float* data = (float*)self.root;

    for(int r =0 ; r < rank-1 ; r++)
        printf("[");

    int keep = 1;
    int jump = 0;
    while(keep)
    {
        printf("[");
        for (int i =0 ; i< shape[rank-1]; i++)
        {
            printf("%.3f,", data[jump + i]);
        }
        jump += shape[rank-1];
        if (jump < self.size)
        {
            printf("]\n");
        }
        else
        {
            printf("]");
            keep = 0;
        }
    }
    for(int r =0 ; r < rank-1 ; r++)
        printf("]\n");
}
void pfprintShape(pf_tensor self)
{
    printf("shape = (");
    for (int i =0 ; i < self.ndim ; i++)
        printf("%d,", self.shape[i]);
    printf(")\n");
}
