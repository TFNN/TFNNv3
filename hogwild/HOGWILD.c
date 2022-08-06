// James William Fletcher (github.com/tfcnn)
// gcc main.c -lm -Ofast -mavx -mfma -o main
// hogwild threaded version, linux/unix only
// sadly this performs worse on a threadripper
// Zen2 than it does an old Zen+ 8 core 2700x.
// also no fma4 on the threadripper!?
//
// Turning off FAST_PREDICTABLE_MODE yeils
// better performance this time as a higher
// entropy random is more important in threaded
// workloads.

#include <sys/time.h>
#include <stdint.h>
#include <unistd.h>
#include <signal.h>

#include <stdatomic.h>
#include <pthread.h>
#include <errno.h>

#include "TFCNNv3_NOBATCHING.h"

#define DS 78   // training/data samples (13 floats per sample 12 input 1 output)
#define DSS 1014// dataset size (num of 4-byte floats)
#define EPOCHS 333333333
#define DEBUG 1
#define NUM_THREADS 16 // change this to the number of cores or threads your CPU has available
f32 dataset[DSS];
network net;

uint kill_threads = 0;
pthread_t tid[NUM_THREADS];
atomic_uint epochs_per_second = 0;

uint64_t microtime()
{
    struct timeval tv;
    struct timezone tz;
    memset(&tz, 0, sizeof(struct timezone));
    gettimeofday(&tv, &tz);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

void run_tests(int sig_num) 
{
    if(sig_num == 2){printf(" Early termination called.\n\n");}

    // gracefully terminate threads
    kill_threads = 1;
    for(int i = 0; i < NUM_THREADS; i++)
    {
        if(pthread_join(tid[i], NULL) != 0)
            printf("failed to join thread %i\n", i);
    }

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

void *train_thread(void *arg)
{
    while(kill_threads == 0)
    {
        for(uint i=0; i < DS; i++)
        {
            const uint ofs = uRand(0, DS-1)*13;
            processNetwork(&net, &dataset[ofs], &dataset[ofs+12], NULL);
        }
        epochs_per_second++;
    }
}

int main()
{
    // ctrl+c callback
    signal(SIGINT, run_tests);

    // force a high priority nice
    errno = 0;
    if(nice(-20) < 0)
    {
        while(errno != 0)
        {
            errno = 0;
            if(nice(-20) < 0)
                printf("Attempting to set process to nice of -20 (run with sudo)...\n");
            sleep(1);
        }
    }

    // load dataset
    FILE* f = fopen("train_xy.dat", "rb");
    if(f != NULL)
    {
        size_t rv = fread(dataset, sizeof(float), DSS, f);
        if(rv != DSS){printf("Reading dataset failed. %li\n", rv); return 1;}
        fclose(f);
    }

    // init network
    int r = createNetwork(&net, WEIGHT_INIT_NORMAL_GLOROT, 12, 1, 3, 32, 1); 
    if(r < 0){printf("Init network failed, error: %i\n", r); return 2;}

    // config network
    setLearningRate(&net, 0.001f);
    setActivator(&net, TANH);
    setOptimiser(&net, OPTIM_ADAGRAD);

    // launch training threads
    for(int i = 0; i < NUM_THREADS; i++)
    {
        if(pthread_create(&tid[i], NULL, train_thread, NULL) != 0)
            printf("failed to create thread %i\n", i);
    }

    // run a training process on the parent too just for the statistics
    uint epoch_seconds = 0;
    for(uint j=0; j < EPOCHS; j++)
    {
        f32 epoch_loss = 0.f;
        for(uint i=0; i < DS; i++)
        {
            const uint ofs = uRand(0, DS-1)*13;
            const f32 loss = processNetwork(&net, &dataset[ofs], &dataset[ofs+12], NULL);
            epoch_loss += loss;
        }

        printf("[%u] epoch loss: %f\n", j, epoch_loss);
        printf("[%u] avg epoch loss: %f\n", j, epoch_loss/DS);

#if DEBUG == 1
        layerStat(&net);
#endif

        epochs_per_second++;
        static uint64_t lt = 0;
        if(microtime() > lt)
        {
            epoch_seconds++;
            lt = microtime()+1000000;
        }
        printf("EPS: %u\n\n", epochs_per_second/epoch_seconds); // epochs per second
    }

    run_tests(0);
    return 0;
}



