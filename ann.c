#include "ann.h"

struct layer *hidden_layer_build(int features, int neurons)
{
        struct layer *l = malloc(sizeof(struct layer));
        l->neurons = neurons;
        l->weights = matrix_alloc(l->neurons, features);
        l->weighted_input = matrix_alloc(l->neurons, 1);
        l->bias = matrix_calloc(l->neurons, l->weighted_input->cols);
        l->activations = matrix_alloc(l->neurons, l->weighted_input->cols);
        float scalar = sqrt(2.0 / features); /* He et al. relu weight initization */
        for(int i = 0; i < l->weights->size; ++i)
                l->weights->data[i] = ((double)rand() / (double)RAND_MAX) * scalar;
        return l;
}

struct layer *output_layer_build(struct ann *a, int neurons)
{ 
        struct layer *l = malloc(sizeof(struct layer));
        l->neurons = neurons;
        l->weights = matrix_calloc(l->neurons, a->hidden->neurons); 
        l->weighted_input = matrix_alloc(l->neurons, 1);
        l->bias = matrix_calloc(l->neurons, l->weighted_input->cols);
        l->activations = l->weighted_input; /* sneaky hobbitses */
        return l;
}

struct gradients *grads_build(struct ann *a, int features)
{
        struct gradients *g = malloc(sizeof(struct gradients));
        g->deltao = a->output->activations; /* aliasanity */
        g->wo_update = matrix_alloc(g->deltao->rows, a->hidden->activations->rows);
        g->deltah = matrix_alloc(a->output->weights->cols, g->deltao->cols);
        g->wh_update = matrix_alloc(g->deltah->rows, features);
        return g;
}

struct ann *ann_build(float regularization, float learn, int observations,
                int hidden, int features, int classes)
{
        srand(time(NULL));
        struct ann *a = malloc(sizeof(struct ann));
        a->hidden = hidden_layer_build(features, hidden);
        a->output = output_layer_build(a, classes);
        a->lrate  = learn;
        a->grads  = grads_build(a, features);
        a->reg = 1.0 - ((a->lrate * regularization) / observations); 
        a->err = 100.0;
        return a;
}

void hidden_layer_free(struct layer *l)
{
        matrix_free(l->activations);
        matrix_free(l->bias);
        matrix_free(l->weighted_input);
        matrix_free(l->weights);
        free(l);
}

void output_layer_free(struct layer *l)
{
        matrix_free(l->bias);
        matrix_free(l->weighted_input);
        matrix_free(l->weights);
        free(l);
}

void grads_free(struct gradients *g)
{
        matrix_free(g->wh_update);
        matrix_free(g->deltah);
        matrix_free(g->wo_update);
        free(g);
}

void ann_free(struct ann *a)
{
        grads_free(a->grads);
        output_layer_free(a->output);
        hidden_layer_free(a->hidden);
        free(a);
}

void ann_learn(struct ann *a, struct iod *io, int epochs)
{
        for(int i = 0; i < io->batch * epochs; ++i)
                ann_train(a, io);
}

void ann_test(struct ann *a, struct iod *io)
{
        int missed = io->batch;
        for(int i = 0; i < io->batch; ++i) {
                ann_fprop(a, io->inputs[i]);
                if(ann_classify(a) == io->c[i])
                        --missed;
        }
        a->err = ((double)missed / (double)io->batch) * 100.0;
}

/*  
 * Did you ever hear the Tragedy of Darth Plagueis the Wise?
 * I thought not. It's not a story the Jedi would tell you.
 * It's a Sith legend. 
 * Darth Plagueis was a Dark Lord of the Sith so powerful and so wise, 
 * he could use the Force to influence the midi-chlorians to create...life. 
 * He had such a knowledge of the Dark Side, 
 * he could even keep the ones he cared about...from dying. 
 * He became so powerful, the only thing he was afraid of was losing his power...
 * which, eventually of course, he did. 
 * Unfortunately, he taught his apprentice everything he knew. 
 * Then his apprentice killed him in his sleep. 
 * Ironic. He could save others from death...but not himself. 
 */
