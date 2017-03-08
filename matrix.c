#include "matrix.h"

struct matrix *matrix_alloc(int rows, int cols)
{
        struct matrix *m = malloc(sizeof(struct matrix));
        m->rows = rows;
        m->cols = cols;
        m->size = m->rows * m->cols;
        m->data = malloc(m->size * sizeof(float));
        return m;
}

struct matrix *matrix_calloc(int rows, int cols)
{
        struct matrix *m = malloc(sizeof(struct matrix));
        m->rows = rows;
        m->cols = cols;
        m->size = m->rows * m->cols;
        m->data = calloc(m->size, sizeof(float));
        return m;
}

void matrix_free(struct matrix *m)
{
        free(m->data);
        free(m);
}

void matrix_add(struct matrix *m, struct matrix *n)
{
        for(int i = 0; i < m->size; ++i)
                m->data[i] += n->data[i];
}

void matrix_sub(struct matrix *m, struct matrix *n)
{
        for(int i = 0; i < m->size; ++i)
                m->data[i] -= n->data[i];
}

void matrix_mul(struct matrix *m, struct matrix *n)
{
        for(int i = 0; i < m->size; ++i)
                m->data[i] *= n->data[i];
}

void matrix_addc(struct matrix *m, float v)
{
        for(int i = 0; i < m->size; ++i)
                m->data[i] += v;
}

void matrix_scale(struct matrix *m, float v)
{
        for(int i = 0; i < m->size; ++i) 
                m->data[i] *= v;
}

float matrix_max(struct matrix *m)
{
        float x;
        float max = -FLT_MAX;
        for(int i = 0; i < m->size; ++i) {
                x = m->data[i];
                if(x > max)
                        max = x;
        }
        return max;
}

int matrix_max_idx(struct matrix *m)
{
        float x;
        float max = -FLT_MAX;
        int j = 0;
        for(int i = 0; i < m->size; ++i) {
                x = m->data[i];
                if(x > max) {
                        max = x;
                        j = i;
                }
        }
        return j;
}
