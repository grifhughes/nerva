#ifndef ANN_H
#define ANN_H

#include "io.h"
#include <mkl.h>
#include <time.h>

struct layer {
        int neurons;
        struct matrix *weights, *bias, *weighted_input, *activations;
};

struct gradients {
        struct matrix *deltao, *wo_update, *deltah, *wh_update;
};

struct ann {
        struct layer *hidden, *output;
        struct gradients *grads;
        float lrate, reg, err;
};

struct layer *hidden_layer_build(int features, int neurons);
struct layer *output_layer_build(struct ann *a, int neurons);
struct gradients *grads_build(struct ann *a, int features);
struct ann *ann_build(float regularization, float learn, int observations, 
                int hidden, int features, int classes);
void hidden_layer_free(struct layer *l);
void output_layer_free(struct layer *l);
void grads_free(struct gradients *g);
void ann_free(struct ann *a);
void ann_learn(struct ann *a, struct iod *io, int epochs);
void ann_test(struct ann *a, struct iod *io);

static inline int ann_classify(struct ann *a)
{
        return matrix_max_idx(a->output->activations);
}

static inline float relu(float x)
{
        return fmaxf(0.0f, x); 
}

static inline void drelu(float *x)
{
        if(*x >= 0.0f) {
                *x = 1.0f;
                return;
        }
        *x = 0.0f;
}

#endif
