#ifndef __PF_BODY_H__
#define __PF_BODY_H__

#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include <string.h>
#include <malloc.h>

#include "nargs.h"
#include "type_definition.h"
#include "device_definition.h"

#define PF_LOG(m) printf("(%s:%d) %s\n", __func__ , __LINE__ , (m))

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
    void*      root;
    PF_TYPE    type;
    
    /* info */ 
    int*       shape;
    int        ndim;
    int        size;
    PF_DEVICE  device;

    /* base operator */
    bool (*mul)(struct pf_tensor* self,       struct pf_tensor* operand, struct pf_tensor* result);
    bool (*add)(struct pf_tensor* self,       struct pf_tensor* operand, struct pf_tensor* result);
    bool (*sub)(struct pf_tensor* self,       struct pf_tensor* operand, struct pf_tensor* result);
    bool (*div)(struct pf_tensor* self,       struct pf_tensor* operand, struct pf_tensor* result);
    bool (*dot)(struct pf_tensor* self,       struct pf_tensor* operand, struct pf_tensor* result);
    bool (*matmul)(struct pf_tensor* self,    struct pf_tensor* operand, struct pf_tensor* result);

    /* func */
    struct pf_tensor (*at)(struct pf_tensor* self, int dim, ...);
    void   (*to)(struct pf_tensor* self, PF_DEVICE device);
    bool   (*set)(struct pf_tensor* self, double value);
    double (*values)(struct pf_tensor* self);
    // bool   (*astype)(struct pf_tensor* self, PF_TYPE type);
}pf_tensor;

typedef bool (*pf_operator)(pf_tensor* self, pf_tensor* operand, pf_tensor* result);

#endif
