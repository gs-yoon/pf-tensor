#include <stdio.h>
#include <stdarg.h> 

int* s3(int p1, int p2, int p3)
{
    int *shape;
    shape[0] = p1;
    shape[1] = p2;
    shape[2] = p3;
    return shape;
}

struct op32f
{
   struct pf_Tensor32f* (*mul)(struct pf_Tensor32f *self,       struct pf_Tensor32f *operand);
//    struct pf_Tensor32f* (*add)(struct pf_Tensor32f *self,       struct pf_Tensor32f *operand);
//    struct pf_Tensor32f* (*sub)(struct pf_Tensor32f *self,       struct pf_Tensor32f *operand);
//    struct pf_Tensor32f* (*div)(struct pf_Tensor32f *self,       struct pf_Tensor32f *operand);
//    struct pf_Tensor32f* (*dot)(struct pf_Tensor32f *self,       struct pf_Tensor32f *operand);
//    struct pf_Tensor32f* (*matMul)(struct pf_Tensor32f *self,    struct pf_Tensor32f *operand);
};

struct pf_info
{
    /* data */
    int*     shape;
    int      ndim;
    int      size;
};


struct pf_Tensor32f
{
    /* data */
    float* root;

    struct pf_info info;
    struct op32f op;

    /* func */
    float (*at)(struct pf_Tensor32f* self, ...);

};

struct pf_Tensor32f* mull(struct pf_Tensor32f *self,       struct pf_Tensor32f *operand)
{
    printf("hi\n");
}


struct Shape
{
    int d;
    int d1;
    int d2;
};
typedef struct Shape Shape;

void testo(int  d,...)
{
    va_list ap;
    va_start(ap, d);
    for (int i = 0; i < d; i++)    // 가변 인자 개수만큼 반복
    {
        int num = va_arg(ap, int);    // int 크기만큼 가변 인자 목록 포인터에서 값을 가져옴
                                      // ap를 int 크기만큼 순방향으로 이동
        printf("%d ", num);           // 가변 인자 값 출력
    }
}

void testt(int s[])
{
    printf("%d\n",s[0]);
    for (int i = 0; i < 3; i++)    // 가변 인자 개수만큼 반복
    {
        printf("%d \n", s[i]);       
    }
}

void init_(struct pf_Tensor32f* t)
{
    t->op.mul = mull;
}
int main()
{
    struct pf_Tensor32f tensor;
    struct pf_Tensor32f a;
    init_(&tensor);
    tensor.op.mul(&tensor, &a);
    Shape d = {2,3,4};
    int* dfd = s3(1,2,3);
    printf("%d\n",dfd[0]);
    testt(dfd);
    // printf("%d\n",stru);
}