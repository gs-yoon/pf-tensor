#ifndef __NARGS_H__
#define __NARGS_H__

/* calculate number of arguments*/
#define _NARGS(_0,_1,_2,_3,_4,_5,_6,N,...) N
#define NARGS(...) _NARGS(__VA_ARGS__,6, 5, 4, 3, 2, 1,0)
#define _CAT(a,b) a ## b
#define CAT(a,b) _CAT(a,b)


/* for pf_tensor.at function*/
#define ATDN(t,p,s,i,d) ATD##d(t,p,s,i, s[d-1], i[d-1])
#define ATD0(t,p) (double)(*((t*)p))
#define ATD1(t,p,s,i,ss,v) (double)(*((t*)p+v))
#define ATD2(t,p,s,i,ss,v) ATD1(t,p,s,i, s[0]*ss, ss*i[0] + v)
#define ATD3(t,p,s,i,ss,v) ATD2(t,p,s,i, s[1]*ss, ss*i[1] + v)
#define ATD4(t,p,s,i,ss,v) ATD3(t,p,s,i, s[2]*ss, ss*i[2] + v)
#define ATD5(t,p,s,i,ss,v) ATD4(t,p,s,i, s[3]*ss, ss*i[3] + v)
#define ATD6(t,p,s,i,ss,v) ATD5(t,p,s,i, s[4]*ss, ss*i[4] + v)
#define ATD7(t,p,s,i,ss,v) ATD6(t,p,s,i, s[5]*ss, ss*i[5] + v)
#define ATD8(t,p,s,i,ss,v) ATD7(t,p,s,i, s[6]*ss, ss*i[6] + v)
#define ATD9(t,p,s,i,ss,v) ATD8(t,p,s,i, s[7]*ss, ss*i[7] + v)


/* for AT(type, pf_tensor,...) function */
#define AT(t,self,...)  CAT(AT,NARGS(0,__VA_ARGS__))(self,t,0,1,__VA_ARGS__)
#define AT1(self,t,v,ss,p1) *((t*)(self)->root + ss*p1 + v)
#define AT2(self,t,v,ss,p1, p2) AT1(self,t,v + ss*p2, ss*(self)->shape[1],p1)
#define AT3(self,t,v,ss,p1, p2,p3) AT2(self,t,v + ss*p3, ss*(self)->shape[2],p1,p2 ) //#
#define AT4(self,t,v,ss,p1, p2,p3,p4) AT3(self,t,v + ss*p4, ss*(self)->shape[3],p1,p2,p3 )
#define AT5(self,t,v,ss,p1, p2,p3,p4,p5) AT4(self,t,v + ss*p5, ss*(self)->shape[4],p1,p2,p3,p4 )
#define AT6(self,t,v,ss,p1, p2,p3,p4,p5,p6) AT5(self,t,v + ss*p6, ss*(self)->shape[5],p1,p2,p3,p4,p5 )
#define AT7(self,t,v,ss,p1, p2,p3,p4,p5,p6,p7) AT6(self,t,v + ss*p7, ss*(self)->shape[6],p1,p2,p3,p4,p5,p6 )
#define AT8(self,t,v,ss,p1, p2,p3,p4,p5,p6,p7,p8) AT7(self,t,v + ss*p8, ss*(self)->shape[7],p1,p2,p3,p4,p5,p6,p7 )
#define AT9(self,t,v,ss,p1, p2,p3,p4,p5,p6,p7,p8,p9) AT8(self,t,v + ss*p9, ss*(self)->shape[8],p1,p2,p3,p4,p5,p6,p7,p8 )

/* for AT(data,shape,...) function */
#define ATS(self,shape,...)  CAT(ATS,NARGS(0,__VA_ARGS__))(self,shape,0,1,__VA_ARGS__)
#define ATS1(self,shape,v,ss,p1) *(self + ss*p1 + v)
#define ATS2(self,shape,v,ss,p1, p2) ATS1(self,shape,v + ss*p2, ss*shape[1],p1)
#define ATS3(self,shape,v,ss,p1, p2,p3) ATS2(self,shape,v + ss*p3, ss*shape[2],p1,p2 ) //#
#define ATS4(self,shape,v,ss,p1, p2,p3,p4) ATS3(self,shape,v + ss*p4, ss*shape[3],p1,p2,p3 )
#define ATS5(self,shape,v,ss,p1, p2,p3,p4,p5) ATS4(self,shape,v + ss*p5, ss*shape[4],p1,p2,p3,p4 )
#define ATS6(self,shape,v,ss,p1, p2,p3,p4,p5,p6) ATS5(self,shape,v + ss*p6, ss*shape[5],p1,p2,p3,p4,p5 )
#define ATS7(self,shape,v,ss,p1, p2,p3,p4,p5,p6,p7) ATS6(self,shape,v + ss*p7, ss*shape[6],p1,p2,p3,p4,p5,p6 )
#define ATS8(self,shape,v,ss,p1, p2,p3,p4,p5,p6,p7,p8) ATS7(self,shape,v + ss*p8, ss*shape[7],p1,p2,p3,p4,p5,p6,p7 )
#define ATS9(self,shape,v,ss,p1, p2,p3,p4,p5,p6,p7,p8,p9) ATS8(self,shape,v + ss*p9, ss*shape[8],p1,p2,p3,p4,p5,p6,p7,p8 )

/* for making tensor, return dimension and shape*/
#define SHAPE(...) CAT(SHAPE,NARGS(0,__VA_ARGS__))(__VA_ARGS__)
#define SHAPE0() 0
#define SHAPE1(p1)  1,p1
#define SHAPE2(p1, p2)  2,p1,p2
#define SHAPE3(p1, p2,p3) 3,p1,p2,p3
#define SHAPE4(p1, p2,p3,p4)  4,p1,p2,p3,p4
#define SHAPE5(p1, p2,p3,p4,p5)  5,p1,p2,p3,p4,p5
#define SHAPE6(p1, p2,p3,p4,p5,p6) 6,p1,p2,p3,p4,p5,p6,
#define SHAPE7(p1, p2,p3,p4,p5,p6,p7) 7,p1,p2,p3,p4,p5,p6,p7,
#define SHAPE8(p1, p2,p3,p4,p5,p6,p7,p8) 8,p1,p2,p3,p4,p5,p6,p7,p8
#define SHAPE9(p1, p2,p3,p4,p5,p6,p7,p8,p9) 9,p1,p2,p3,p4,p5,p6,p7,p8,p9


#define VA_IDX(d, p)        \
    va_list ap;              \
    va_start(ap, d);          \
    for (int i = 0; i < d; i++)\
        p[i] = va_arg(ap, int); \
	va_end(ap);                  

#endif