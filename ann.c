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
                l->weights->data[i] = ((float)rand() / (float)RAND_MAX) * scalar;
        return l;
}

struct layer *output_layer_build(struct ann *a, int neurons)
{ 
        struct layer *l = malloc(sizeof(struct layer));
        l->neurons = neurons;
        l->weights = matrix_calloc(l->neurons, a->hidden->neurons); 
        l->weighted_input = matrix_alloc(l->neurons, 1);
        l->bias = matrix_calloc(l->neurons, l->weighted_input->cols);
        l->activations = l->weighted_input; /* sneaky */
        /* output layer's weighted input is not used to compute the gradients,
         * therefore it can safely be aliased to avoid an extra allocation */ 
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
        a->grads  = grads_build(a, features);
        a->lrate  = learn;
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

static void softmax(struct matrix *in)
{
        float x;
        float sum = 0.0f;
        float max = -matrix_max(in);
        matrix_addc(in, max);
        for(int i = 0; i < in->size; ++i) {
                x = exp(in->data[i]);
                in->data[i] = x;
                sum += x;
        }
        sum = 1.0f / sum;
        matrix_scale(in, sum);
}

static void ann_fprop(struct ann *a, struct matrix *example) 
{
        /* input -> hidden */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        a->hidden->weights->rows, example->cols, a->hidden->weights->cols, 
                        1.0, a->hidden->weights->data, a->hidden->weights->rows, 
                        example->data, example->rows, 0.0, 
                        a->hidden->weighted_input->data, a->hidden->weighted_input->rows);
        matrix_add(a->hidden->weighted_input, a->hidden->bias);
        for(int i = 0; i < a->hidden->activations->size; ++i)
                a->hidden->activations->data[i] = relu(a->hidden->weighted_input->data[i]);

        /* hidden -> output */
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        a->output->weights->rows, a->hidden->activations->cols, a->output->weights->cols,
                        1.0, a->output->weights->data, a->output->weights->rows,
                        a->hidden->activations->data, a->hidden->activations->rows, 0.0,
                        a->output->weighted_input->data, a->output->weighted_input->rows);
        matrix_add(a->output->weighted_input, a->output->bias);
        softmax(a->output->activations);
}

static void ann_bprop(struct ann *a, struct matrix *example, struct matrix *target)
{
        /* deltao & de w.r.t output weights */
        matrix_sub(a->grads->deltao, target);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, 
                        a->grads->deltao->rows, a->hidden->activations->rows, a->grads->deltao->cols,
                        1.0, a->grads->deltao->data, a->grads->deltao->rows,
                        a->hidden->activations->data, a->hidden->activations->rows, 0.0,
                        a->grads->wo_update->data, a->grads->wo_update->rows);

        /* deltah & de w.r.t hidden weights */
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        a->output->weights->cols, a->grads->deltao->cols, a->output->weights->rows,
                        1.0, a->output->weights->data, a->output->weights->rows,
                        a->grads->deltao->data, a->grads->deltao->rows, 0.0,
                        a->grads->deltah->data, a->grads->deltah->rows);
        for(int i = 0; i < a->hidden->weighted_input->size; ++i)
                drelu(&a->hidden->weighted_input->data[i]);
        matrix_mul(a->grads->deltah, a->hidden->weighted_input);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        a->grads->deltah->rows, example->rows, a->grads->deltah->cols,
                        1.0, a->grads->deltah->data, a->grads->deltah->rows,
                        example->data, example->rows, 0.0,
                        a->grads->wh_update->data, a->grads->wh_update->rows); 
}

static void ann_update_weights(struct ann *a)
{
        /* the real bottleneck */
        matrix_scale(a->grads->deltao, a->lrate);
        matrix_scale(a->grads->wo_update, a->lrate);
        matrix_scale(a->grads->deltah, a->lrate);
        matrix_scale(a->grads->wh_update, a->lrate);
        matrix_scale(a->output->weights, a->reg);
        matrix_scale(a->hidden->weights, a->reg);

        matrix_sub(a->output->bias, a->grads->deltao);
        matrix_sub(a->output->weights, a->grads->wo_update);
        matrix_sub(a->hidden->bias, a->grads->deltah);
        matrix_sub(a->hidden->weights, a->grads->wh_update);
}

static void ann_train(struct ann *a, struct iod *io)
{
        /* randomly sampling a training example each iteration adds noise to
         * training */
        unsigned ind = ((io->batch - 1) + 1) * ((float)rand() / (float)RAND_MAX);
        ann_fprop(a, io->inputs[ind]);
        ann_bprop(a, io->inputs[ind], io->targets[ind]);
        ann_update_weights(a);
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
        a->err = ((float)missed / (float)io->batch) * 100.0f;
}
