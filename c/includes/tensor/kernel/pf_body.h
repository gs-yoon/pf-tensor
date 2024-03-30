#ifndef __PF_BODY_H__
#define __PF_BODY_H__

#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include <string.h>
#include <malloc.h>

#include "nargs.h"
#include "type_definition.h"

typedef enum PF_TYPE {
    PF_UNDEFIEND = 0,
    PF_INT8, //1
    PF_INT32,
    PF_UINT8,
    PF_UINT32,
    PF_FLOAT32,
    PF_FLOAT64,
    }PF_TYPE;


typedef struct pf_info
{
    /* data */
    int*     shape;
    int      ndim;
    int      size;
}pf_info;

typedef struct pf_tensor
{
    /* data */
    void*    root;
    PF_TYPE  type;
    
    /* info */ 
    int*     shape;
    int      ndim;
    int      size;

    /* base operator */
    struct pf_tensor* (*mul)(struct pf_tensor* self,       struct pf_tensor* operand);
    struct pf_tensor* (*add)(struct pf_tensor* self,       struct pf_tensor* operand);
    struct pf_tensor* (*sub)(struct pf_tensor* self,       struct pf_tensor* operand);
    struct pf_tensor* (*div)(struct pf_tensor* self,       struct pf_tensor* operand);
    struct pf_tensor* (*dot)(struct pf_tensor* self,       struct pf_tensor* operand);
    struct pf_tensor* (*matMul)(struct pf_tensor* self,    struct pf_tensor* operand);

    /* func */
    double (*at)(struct pf_tensor* self, int dim, ...);
}pf_tensor;

#endif
