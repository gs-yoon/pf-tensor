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
// #define PF_LOG(fmt, ...) \
    // printf("[%s: %d][%s] " fmt "\t\t\t (%s, %s)\n", \
    // __FILE__, __LINE__, __func__, __DATE__, __TIME__);

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
    double (*at)(struct pf_tensor* self, int dim, ...);
    void   (*to)(struct pf_tensor* self, PF_DEVICE device);
    bool   (*set)(struct pf_tensor* self, double value);
    // bool   (*astype)(struct pf_tensor* self, PF_TYPE type);
}pf_tensor;

#endif
