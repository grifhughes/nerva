#ifndef MATRIX_H
#define MATRIX_H

#include <float.h>
#include <stdlib.h>

/* column major storage to avoid internal copying in blas */
struct matrix {
        int rows, cols, size; 
        float *data;
};

struct matrix *matrix_alloc(int rows, int cols);
struct matrix *matrix_calloc(int rows, int cols);
void matrix_add(struct matrix *m, struct matrix *n);
void matrix_sub(struct matrix *m, struct matrix *n);
void matrix_mul(struct matrix *m, struct matrix *n);
void matrix_addc(struct matrix *m, float v);
void matrix_scale(struct matrix *m, float v);
void matrix_free(struct matrix *m);
float matrix_max(struct matrix *m);
int matrix_max_idx(struct matrix *m);

static inline float matrix_get(struct matrix *m, int i, int j)
{
        return m->data[m->rows * j + i]; 
}

static inline void matrix_set(struct matrix *m, int i, int j, float v)
{
        m->data[m->rows * j + i] = v;
}

#endif
