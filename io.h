#ifndef IO_H
#define IO_H

#include "matrix.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* stores parsed training and testing data from .csv */
struct iod {
        int batch;
        int *c;
        struct matrix **inputs, **targets; 
};

struct iod *iod_alloc(int batch, int features, int classes);
void iod_free(struct iod *io);
void iod_parse(struct iod *io, FILE *fp);

#endif
