#include "ann.h"

#define TRSIZE 60000 
#define TESIZE 10000
#define REGULARIZATION 0.5
#define HIDDEN 200 
#define FEATURES 784
#define CLASSES 10
#define LEARN 0.005
#define EPOCHS 35 

int
main(void)
{
        FILE *trf, *tef;
        struct iod *train, *test;
        struct ann *a;

        trf = fopen("mnist_train.csv", "r");
        tef = fopen("mnist_test.csv", "r");
        train = iod_alloc(TRSIZE, FEATURES, CLASSES);
        test = iod_alloc(TESIZE, FEATURES, CLASSES);
        iod_parse(train, trf);
        iod_parse(test, tef);

        a = ann_build(REGULARIZATION, LEARN, TRSIZE, HIDDEN, FEATURES, CLASSES);
        ann_learn(a, train, EPOCHS);
        ann_test(a, test);
        printf("err%%: %.2lf\n", a->err);
    
        ann_free(a);
        iod_free(test);
        iod_free(train);
}
