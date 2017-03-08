#include "io.h"

#define BUF_LEN 4096 

struct iod *iod_alloc(int size, int features, int classes)
{
        struct iod *tmp = malloc(sizeof(struct iod));
        tmp->batch = size;
        tmp->c = malloc(tmp->batch * sizeof(int));
        tmp->inputs = malloc(tmp->batch * sizeof(struct matrix *));
        tmp->targets = malloc(tmp->batch * sizeof(struct matrix *));
        for(int i = 0; i < tmp->batch; ++i) {
                tmp->inputs[i] = matrix_alloc(features, 1);
                tmp->targets[i] = matrix_calloc(classes, 1); 
        }
        return tmp;
}

void iod_free(struct iod *io)
{
        for(int i = 0; i < io->batch; ++i) {
                matrix_free(io->targets[i]);
                matrix_free(io->inputs[i]);
        }
        free(io->targets);
        free(io->inputs);
        free(io->c);
        free(io);
}

void iod_parse(struct iod *io, FILE *fp)
{
        char *l;
        char line[BUF_LEN];
        float sum, x, mean, sdev_sum, stddev;
        for(int i = 0; i < io->batch; ++i) {
                fgets(line, BUF_LEN, fp);
                l = strtok(line, ",");
                io->c[i] = atoi(l);
                matrix_set(io->targets[i], io->c[i], 0, 1.0);

                /* parse & perform gaussian normalization */
                sum = 0.0f;
                for(int j = 0; j < io->inputs[i]->rows; ++j) {
                        x = atof(l);
                        matrix_set(io->inputs[i], j, 0, x);
                        sum += x;
                        l = strtok(NULL, ",");
                }
                mean = sum / io->inputs[i]->rows;
                sdev_sum = 0.0f;
                for(int k = 0; k < io->inputs[i]->rows; ++k) {
                        x = matrix_get(io->inputs[i], k, 0) - mean;
                        matrix_set(io->inputs[i], k, 0, x);
                        sdev_sum += (x * x);
                }
                stddev = sqrt(sdev_sum / io->inputs[i]->rows);
                for(int g = 0; g < io->inputs[i]->rows; ++g) {
                        x = matrix_get(io->inputs[i], g, 0) / stddev;
                        matrix_set(io->inputs[i], g, 0, x);
                }
        }
}
