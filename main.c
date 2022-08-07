// James William Fletcher (github.com/tfcnn)
// gcc main.c -lm -Ofast -mavx -mfma -o main

#include <stdint.h>
#include <unistd.h>
#include <signal.h>

#include "TFCNNv3.h"

#ifdef __linux__
    #include <sys/time.h>
#elif
    #include <time.h>
#endif

#define DS 78   // training/data samples (13 floats per sample 12 input 1 output)
#define DSS 1014// dataset size (num of 4-byte floats)
#define EPOCHS 333333333
#define DEBUG 1
f32 dataset[DSS];
network net;

uint64_t microtime()
{
    struct timeval tv;
    struct timezone tz;
    memset(&tz, 0, sizeof(struct timezone));
    gettimeofday(&tv, &tz);
    return 1000000 * tv.tv_sec + tv.tv_usec;
    
    // struct timespec ts;
    // clock_gettime(CLOCK_MONOTONIC, &ts);
    // return (uint64_t) ts.tv_sec * (uint64_t) 1000000 + (uint64_t) (ts.tv_nsec / 1000);
}

void shuffle_dataset()
{
    for(int i = 0; i < DS; i++)
    {
        const int i1 = uRand(0, DS-2);
        int i2 = i1;
        while(i1 == i2)
            i2 = uRand(0, DS-2);
        f32 t[13];
        memcpy(&t, &dataset[i1*13], 13*sizeof(f32));
        memcpy(&dataset[i1*13], &dataset[i2*13], 13*sizeof(f32));
        memcpy(&dataset[i2*13], t, 13*sizeof(f32));
    }
}

void run_tests(int sig_num) 
{
    if(sig_num == 2){printf(" Early termination called.\n\n");}

    f32 input[12];
    f32 output = 0.f;

    // random tests
    printf("Random Inputs Test:\n");
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 12; j++)
            input[j] = uRandFloat(0.f, 3.f);
        processNetwork(&net, &input[0], NULL, &output);
        printf("%.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f: %.2f%%\n", input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8], input[9], input[10], input[11], output);
    }

    // reload the dataset so that it's unshuffled..
    FILE* f = fopen("train_xy.dat", "rb");
    if(f != NULL)
    {
        size_t rv = fread(dataset, sizeof(float), DSS, f);
        if(rv != DSS){printf("Reading dataset failed. %li\n", rv); exit(0);}
        fclose(f);
    }

    // dataset test
    printf("\nDataset Test:\n");
    for(uint i=0; i < DS; i++)
    {
        const uint ofs = i*13;
        processNetwork(&net, &dataset[ofs], NULL, &output);
        printf("%.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f%.0f: %.2f%%\n", dataset[ofs], dataset[ofs+1], dataset[ofs+2], dataset[ofs+3], dataset[ofs+4], dataset[ofs+5], dataset[ofs+6], dataset[ofs+7], dataset[ofs+8], dataset[ofs+9], dataset[ofs+10], dataset[ofs+11], output);
    }
    
    // save network
    //saveNetwork(&net, "network.save");

    // done
    destroyNetwork(&net);
    exit(0);
}

int main()
{
    // ctrl+c callback
    signal(SIGINT, run_tests);

    // load dataset
    FILE* f = fopen("train_xy.dat", "rb");
    if(f != NULL)
    {
        size_t rv = fread(dataset, sizeof(float), DSS, f);
        if(rv != DSS){printf("Reading dataset failed. %li\n", rv); return 1;}
        fclose(f);
    }
    else
    {
        printf("train_xy.dat not found.\n");
        return 1;
    }
    
    // init network
    int r = createNetwork(&net, WEIGHT_INIT_NORMAL_GLOROT, 12, 1, 3, 32, 1); 
    if(r < 0){printf("Init network failed, error: %i\n", r); return 2;}

    // config network
    // setWeightInit(&net, WEIGHT_INIT_NORMAL_GLOROT);
    // setGain(&net, 1.f);
    // setUnitDropout(&net, 0.f);
    setLearningRate(&net, 0.001f);
    setActivator(&net, TANH);
    setOptimiser(&net, OPTIM_ADAGRAD);
    setBatches(&net, 1);
    
    // setLearningRate(&net, 0.1f);
    // const f32 lr_rr = 0.09999f / (f32)EPOCHS; // learning rate _ reduction rate

    // train network
    uint epochs_per_second = 0;
    uint epoch_seconds = 0;
    for(uint j=0; j < EPOCHS; j++)
    {
        f32 epoch_loss = 0.f;
        for(uint i=0; i < DS; i++)
        {
            const uint ofs = i*13;
            const f32 loss = processNetwork(&net, &dataset[ofs], &dataset[ofs+12], NULL);
            epoch_loss += loss;
            //printf("[%u] loss: %f\n", j+i, loss);
        }
        shuffle_dataset();

        printf("[%u] epoch loss: %f\n", j, epoch_loss);
        printf("[%u] avg epoch loss: %f\n", j, epoch_loss/DS);
        
        // const f32 nlr = 0.1f-(lr_rr*(f32)j);
        // setLearningRate(&net, nlr);
        // printf("LR: %g\n", nlr);

#if DEBUG == 1
        layerStat(&net);
#endif

        // just a test to see how accurate time(0) is at measuring seconds, not good.
        // static uint64_t mt = 0;
        // static uint64_t ltm = 0;
        // epochs_per_second++;
        // static time_t lt = 0;
        // if(time(0) > lt)
        // {
        //     epoch_seconds++;
        //     mt = microtime()-ltm;
        //     ltm = microtime();
        //     lt = time(0)+1;
        // }
        // printf("EPS: %u %lu\n\n", epochs_per_second/epoch_seconds, mt);

#ifdef __linux__
        epochs_per_second++;
        static uint64_t lt = 0;
        if(microtime() > lt)
        {
            epoch_seconds++;
            lt = microtime()+1000000;
        }
        printf("EPS: %u\n\n", epochs_per_second/epoch_seconds); // epochs per second
#elif
        epochs_per_second++;
        static time_t lt = 0;
        if(time(0) > lt)
        {
            epoch_seconds++;
            lt = time(0)+1;
        }
        printf("EPS: %u %lu\n\n", epochs_per_second/epoch_seconds, mt);
#endif
    }

    // training done let's run some basic validation forward passes.
    run_tests(0);
    return 0;
}



