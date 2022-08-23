/*
--------------------------------------------------
    James William Fletcher (github.com/tfnn)
        July 2022 - TFNNv3 (v3.04)
--------------------------------------------------
    
    Tiny Fully Connected Neural Network Library
    https://github.com/tfnn

    Pass target_outputs as NULL to processNetwork()
    for a forward pass / no training.

    Turning off FAST_PREDICTABLE_MODE will use
    the platform dependent (Linux/Unix) /dev/urandom
    it's two times slower but has higher entropy.

    Turning off NOSSE will enable the use of SSE for
    square roots, which actually can be slower.

    - Why no softmax?
    Softmax is targeted more towards classification of
    categories, which is better in a CNN where your outputs
    are onehot vector category classes. Here we have linear
    output layers because they fit well to a wider range of
    applications.

    - Why no ADAM optimiser?
    Requires an extra parameter per weight, too much memory
    bandwidth usage over ADAGRAD.

    This is what I deem to be the best feature set for a
    FNN that will execute fast and efficiently on a CPU
    while still retaining a clean and portable codebase.
    
    You may want to compile this with the -mavx -mfma
    flags for automatic vectorisation.
    
*/

#ifndef TFNN_H
#define TFNN_H

#include <stdio.h>  // fopen, fclose, fwrite, fread, printf
#include <stdlib.h> // malloc, free, exit
#include <math.h>   // tanhf, fabsf, expf, powf, sqrtf, logf, roundf
#include <string.h> // memset, memcpy

#define FAST_PREDICTABLE_MODE
#define NOSSE
#define VERBOSE

#ifndef NOSSE
    #include <x86intrin.h>
#endif

#ifndef FAST_PREDICTABLE_MODE
    #include <sys/file.h>
#endif

#define f32 float
#define uint unsigned int
#define forceinline __attribute__((always_inline)) inline

/*
--------------------------------------
    structures
--------------------------------------
*/

// perceptron struct
struct
{
    f32* data;
    f32* momentum;
    f32 bias;
    f32 bias_momentum;
    uint weights;
}
typedef ptron;

// network struct
struct
{
    // hyperparameters
    uint init;
    uint activator;
    uint optimiser;
    uint batches;
    f32  rate;
    f32  gain;
    f32  dropout;
    f32  wdropout;
    f32  dropout_decay;
    f32  momentum;
    f32  rmsalpha;
    f32  elualpha;
    f32  epsilon;

    // layers
    ptron** layer;

    // count
    uint num_inputs;
    uint num_outputs;
    uint num_layers;
    uint num_layerunits;

    // batch accumulation backprop
    uint  cbatches;// batch iteration counter
    f32** output;  // unit outputs for backprop
    f32*  error;   // total error of each output
    f32*  foutput; // array of outputs
}
typedef network;

/*
--------------------------------------
    ERROR TYPES
--------------------------------------
*/

#define ERROR_UNINITIALISED_NETWORK -1
#define ERROR_TOOFEWINPUTS -2
#define ERROR_TOOFEWLAYERS -3
#define ERROR_TOOSMALL_LAYERSIZE -4
#define ERROR_ALLOC_LAYERS_ARRAY_FAIL -5
#define ERROR_ALLOC_LAYERS_FAIL -6
#define ERROR_ALLOC_OUTPUTLAYER_FAIL -7
#define ERROR_ALLOC_PERCEPTRON_DATAWEIGHTS_FAIL -8
#define ERROR_ALLOC_PERCEPTRON_ALPHAWEIGHTS_FAIL -9
#define ERROR_CREATE_FIRSTLAYER_FAIL -10
#define ERROR_CREATE_HIDDENLAYER_FAIL -11
#define ERROR_CREATE_OUTPUTLAYER_FAIL -12
#define ERROR_ALLOC_OUTPUT_ARRAY_FAIL -13
#define ERROR_ALLOC_OUTPUT_FAIL -14
#define ERROR_ALLOC_BATCH_OUTPUT_FAIL -15
#define ERROR_ALLOC_BATCH_ERROR_FAIL -16
#define ERROR_BAD_FUNCTION_PARAMETERS -17

/*
--------------------------------------
    DEFINES / ENUMS
--------------------------------------
*/

enum 
{
    WEIGHT_INIT_UNIFORM             = 0,
    WEIGHT_INIT_UNIFORM_GLOROT      = 1,
    WEIGHT_INIT_UNIFORM_LECUN       = 2,
    WEIGHT_INIT_UNIFORM_LECUN_POW   = 3,
    WEIGHT_INIT_UNIFORM_RELU        = 4, // he initialisation
    WEIGHT_INIT_NORMAL              = 5,
    WEIGHT_INIT_NORMAL_GLOROT       = 6,
    WEIGHT_INIT_NORMAL_LECUN        = 7,
    WEIGHT_INIT_NORMAL_LECUN_POW    = 8,
    WEIGHT_INIT_NORMAL_RELU         = 9  // he initialisation
}
typedef weight_init_type;

enum 
{
    IDENTITY    = 0,
    ATAN        = 1,
    TANH        = 2,
    ELU         = 3,
    LEAKYRELU   = 4,
    RELU        = 5,
    SIGMOID     = 6,
    SWISH       = 7,
    LECUN       = 8,
    ELLIOT      = 9, // also known as softsign
    SOFTPLUS    = 10,
}
typedef activator;

enum 
{
    OPTIM_SGD       = 0,
    OPTIM_MOMENTUM  = 1,
    OPTIM_NESTEROV  = 2,
    OPTIM_ADAGRAD   = 3,
    OPTIM_RMSPROP   = 4
}
typedef optimiser;

/*
--------------------------------------
    random functions
--------------------------------------
*/

// SSE sqrt
forceinline f32 sqrtps(f32 f)
{
#ifdef NOSSE
    return sqrtf(f);
#else
    return _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(f)));
#endif
}

forceinline f32 rsqrtss(f32 f)
{
#ifdef NOSSE
    return 1.f/sqrtf(f);
#else
    return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(f)));
#endif
}

// random functions
f32  uRandNormal();
f32  uRandFloat(const f32 min, const f32 max);
f32  uRandWeight(const f32 min, const f32 max);
uint uRand(const uint min, const uint umax);
void srandf(const int seed);

/*
--------------------------------------
    accessors
--------------------------------------
*/

void setWeightInit(network* net, const weight_init_type u);
void setOptimiser(network* net, const optimiser u);
void setActivator(network* net, const activator u);
void setBatches(network* net, const uint u);
void setLearningRate(network* net, const f32 f);
void setGain(network* net, const f32 f);
void setUnitDropout(network* net, const f32 f);   // dropout
void setWeightDropout(network* net, const f32 f); // drop connect
void setDropoutDecay(network* net, const f32 f);  // set dropout to silence the unit activation by decay rather than on/off
void setMomentum(network* net, const f32 f); // MOMENTUM & NESTEROV
void setRMSAlpha(network* net, const f32 f); // RMSPROP
void setELUAlpha(network* net, const f32 f); // ELU & LeakyReLU
void setEpsilon(network* net, const f32 f);  // ADAGRAD & RMSPROP
void randomHyperparameters(network* net);

/*
--------------------------------------
    neural net functions
--------------------------------------
*/

int  createNetworkOptimalSmall(network* net, const uint num_inputs, const uint num_outputs);    // ~8.2 KiB
int  createNetworkOptimal(network* net, const uint num_inputs, const uint num_outputs);         // ~28.3 KiB
int  createNetworkOptimalAccurate(network* net, const uint num_inputs, const uint num_outputs); // ~36.6 KiB

int  createNetwork(network* net, const weight_init_type init_type, const uint num_inputs, const uint num_outputs, const uint num_hidden_layers, const uint num_layer_units, const uint default_settings);
f32  processNetwork(network* net, const f32* inputs, const f32* target_outputs, f32* outputs);
void resetNetwork(network* net);
void destroyNetwork(network* net);
int  saveNetwork(network* net, const char* file);
int  loadNetwork(network* net, const char* file);

// a method of getting a concise overview of network weights
void layerStat(network* net);

/*
--------------------------------------
    the code ...
--------------------------------------
*/

static int srandfq = 1988;
void srandf(const int seed)
{
    srandfq = seed;
}

static inline f32 urandf() // 0 to 1
{
#ifdef FAST_PREDICTABLE_MODE
    // https://www.musicdsp.org/en/latest/Other/273-fast-float-random-numbers.html
    // moc.liamg@seir.kinimod
    srandfq *= 16807;
    return (f32)(srandfq & 0x7FFFFFFF) * 4.6566129e-010f;
#else
    static const f32 FLOAT_UINT64_MAX = (f32)UINT64_MAX;
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    ssize_t result = read(f, &s, sizeof(uint64_t));
    close(f);
    return (((f32)s)+1e-7f) / FLOAT_UINT64_MAX;
#endif
}

static inline f32 urandfc() // -1 to 1
{
#ifdef FAST_PREDICTABLE_MODE
    // https://www.musicdsp.org/en/latest/Other/273-fast-float-random-numbers.html
    // moc.liamg@seir.kinimod
    srandfq *= 16807;
    return ((f32)srandfq) * 4.6566129e-010f;
#else
    static const f32 FLOAT_UINT64_MAX_HALF = (f32)(UINT64_MAX/2);
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    ssize_t result = read(f, &s, sizeof(uint64_t));
    close(f);
    return ((((f32)s)+1e-7f) / FLOAT_UINT64_MAX_HALF) - 1.f;
#endif
}

forceinline f32 uRandFloat(const f32 min, const f32 max)
{
    return ( urandf() * (max-min) ) + min;
}

f32 uRandWeight(const f32 min, const f32 max)
{
    f32 pr = 0;
    while(pr == 0) //never return 0
    {
        const f32 rv2 = ( urandf() * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
}

forceinline uint uRand(const uint min, const uint max)
{
    return ( urandf() * (max-min) ) + min;
}

f32 uRandNormal()
{
    // Box-Muller transform: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    f32 u = urandfc();
    f32 v = urandfc();
    f32 r = u * u + v * v;
    while(r == 0.f || r > 1.f)
    {
        u = urandfc();
        v = urandfc();
        r = u * u + v * v;
    }
    return u * sqrtps(-2.f * logf(r) / r);
}

/**********************************************/

forceinline f32 softplus(const f32 x) //derivative is sigmoid()
{
    return logf(1.f + expf(x));
}

forceinline f32 atanDerivative(const f32 x) //atanf()
{
    return 1.f / (1.f + (x*x));
}

forceinline f32 tanhDerivative(const f32 x) //tanhf()
{
    return 1.f - (x*x);
}

/**********************************************/

forceinline f32 elu(const network* net, const f32 x)
{
    //if(x < 0.f){return 1.0507f * (1.67326f * expf(x) - 1.67326f);} // paper claimed these constants?
    if(x < 0.f){return 1.0507f * 1.67326f * (expf(x) - 1.f);} // keras claims this is correct?
    return x;
}

forceinline f32 eluDerivative(const network* net, const f32 x)
{
    if(x > 0.f){return 1.f;}
    return net->elualpha * expf(x);
}

/**********************************************/

forceinline f32 leaky_relu(const network* net, const f32 x)
{
    if(x < 0.f){return x * net->elualpha;}
    return x;
}

forceinline f32 leaky_reluDerivative(const network* net, const f32 x)
{
    if(x > 0.f){return 1.f;}
    return net->elualpha;
}

/**********************************************/

forceinline f32 relu(const f32 x)
{
    if(x < 0.f){return 0.f;}
    return x;
}

forceinline f32 reluDerivative(const f32 x)
{
    if(x > 0.f){return 1.f;}
    return 0.f;
}

/**********************************************/

forceinline f32 swish(const f32 x)
{
    return x / (1.f + expf(-x));
}

forceinline f32 swishDerivative(const f32 x)
{
    const f32 ex = expf(-x);
    const f32 oex = 1.f + ex;
    return 1.f + ex + x * ex / (oex * oex);
}

/**********************************************/

forceinline f32 sigmoid(const f32 x)
{
    return 1.f / (1.f + expf(-x));
}

forceinline f32 sigmoidDerivative(const f32 x)
{
    return x * (1.f - x);
}

/**********************************************/

forceinline f32 elliot_sigmoid(const f32 x) // aka softsign
{
    return x / (1.f + fabsf(x));
}

forceinline f32 elliot_sigmoidDerivative(const f32 x)
{
    const f32 a = 1.f - fabsf(x);
    return a*a;
}

/**********************************************/

forceinline f32 lecun_tanh(const f32 x)
{
    return 1.7159f * tanhf(0.666666667f * x);
}

forceinline f32 lecun_tanhDerivative(const f32 x)
{
    // this is a close enough approximation that
    // I literally "felt" the values for until
    // it seemed about as suitable as I could
    // get it to; 1.1439288854598999023
    // the maximum deviance of this function
    // against the proposed solution;
    // 1.14393 * pow(1 / cosh(x * 0.666667), 2)
    // is; 0.0000011390194490 at x = 0
    // Alternate form as suggested by Wolfram:
    // -0.388522 * (-1.7159 + x) * (1.7159 + x);

    // const f32 sx = x * 0.62331494f;
    // return 1.1439288854598999023f-(sx*sx);

    const f32 sx = x * 0.6233149171f;
    return 1.143928885f-(sx*sx);
}

/**********************************************/

static inline f32 Derivative(const f32 x, const network* net)
{
    if(net->activator == 1)
        return atanDerivative(x);
    else if(net->activator == 2)
        return tanhDerivative(x);
    else if(net->activator == 3)
        return eluDerivative(net, x);
    else if(net->activator == 4)
        return leaky_reluDerivative(net, x);
    else if(net->activator == 5)
        return reluDerivative(x);
    else if(net->activator == 6)
        return sigmoidDerivative(x);
    else if(net->activator == 7)
        return swishDerivative(x);
    else if(net->activator == 8)
        return lecun_tanhDerivative(x);
    else if(net->activator == 9)
        return elliot_sigmoidDerivative(x);
    else if(net->activator == 10)
        return sigmoid(x); // this is the derivative of softplus
    
    return reluDerivative(x); // same as identity derivative
}

static inline f32 Activator(const f32 x, const network* net)
{
    if(net->activator == 1)
        return atanf(x);
    else if(net->activator == 2)
        return tanhf(x);
    else if(net->activator == 3)
        return elu(net, x);
    else if(net->activator == 4)
        return leaky_relu(net, x);
    else if(net->activator == 5)
        return relu(x);
    else if(net->activator == 6)
        return sigmoid(x);
    else if(net->activator == 7)
        return swish(x);
    else if(net->activator == 8)
        return lecun_tanh(x);
    else if(net->activator == 9)
        return elliot_sigmoid(x);
    else if(net->activator == 10)
        return softplus(x);

    return x;
}

/**********************************************/

forceinline f32 SGD(network* net, const f32 input, const f32 error)
{
    return net->rate * error * input;
}

forceinline f32 Momentum(network* net, const f32 input, const f32 error, f32* momentum)
{
    const f32 err = (net->rate * error * input) + net->momentum * momentum[0];
    momentum[0] = err;
    return err;
}

forceinline f32 Nesterov(network* net, const f32 input, const f32 error, f32* momentum)
{
    const f32 v = net->momentum * momentum[0] + ( net->rate * error * input );
    const f32 n = v + net->momentum * (v - momentum[0]);
    momentum[0] = v;
    return n;
}

forceinline f32 ADAGrad(network* net, const f32 input, const f32 error, f32* momentum)
{
    const f32 err = error * input;
    momentum[0] += err * err;
#ifdef NOSSE
    return (net->rate / sqrtf(momentum[0] + net->epsilon)) * err;
#else
    return (net->rate * rsqrtss(momentum[0] + net->epsilon)) * err;
#endif
}

forceinline f32 RMSProp(network* net, const f32 input, const f32 error, f32* momentum)
{
    const f32 err = error * input;
    momentum[0] = net->rmsalpha * momentum[0] + (1.f - net->rmsalpha) * (err * err);
#ifdef NOSSE
    return (net->rate / sqrtf(momentum[0] + net->epsilon)) * err;
#else
    return (net->rate * rsqrtss(momentum[0] + net->epsilon)) * err;
#endif
}

static inline f32 Optimiser(network* net, const f32 input, const f32 error, f32* momentum)
{
    if(net->optimiser == 1)
        return Momentum(net, input, error, momentum);
    else if(net->optimiser == 2)
        return Nesterov(net, input, error, momentum);
    else if(net->optimiser == 3)
        return ADAGrad(net, input, error, momentum);
    else if(net->optimiser == 4)
        return RMSProp(net, input, error, momentum);
    
    return SGD(net, input, error);
}

/**********************************************/

forceinline f32 doPerceptron(const f32* in, ptron* p)
{
    f32 ro = 0.f;
    for(uint i = 0; i < p->weights; i++)
        ro += in[i] * p->data[i]; // descend the weights by gradient/error
    ro += p->bias; // descend the bias by gradient/error
    return ro;
}

static inline f32 doDropout(const network* net, const f32 f, const uint type)
{
    if(type == 1) // unit dropout
    {
        return f * (1.0f - net->dropout_decay);
    }
    else if(type == 2) // weight dropout
    {
        if(uRandFloat(0.f, 1.f) <= net->wdropout)
        {
            if(net->dropout_decay != 0.f)
                return f * (1.0f - net->dropout_decay);
            else
                return 0.f;
        }
    }
    return f;
}

/**********************************************/

int createPerceptron(ptron* p, const uint weights, const f32 d, const weight_init_type wit)
{
    p->data = malloc(weights * sizeof(f32));
    if(p->data == NULL)
        return ERROR_ALLOC_PERCEPTRON_DATAWEIGHTS_FAIL;

    p->momentum = malloc(weights * sizeof(f32));
    if(p->momentum == NULL)
    {
        free(p->data);
        return ERROR_ALLOC_PERCEPTRON_ALPHAWEIGHTS_FAIL;
    }

    p->weights = weights;

    for(uint i = 0; i < p->weights; i++)
    {
        if(wit < 5)
            p->data[i] = uRandWeight(-1.f, 1.f) * d; // uniform
        else
            p->data[i] = uRandNormal() * d; // normal

        p->momentum[i] = 0.f;
    }

    p->bias = 0.f;
    p->bias_momentum = 0.f;

    return 0;
}

void resetPerceptron(ptron* p, const f32 d, const weight_init_type wit)
{
    for(uint i = 0; i < p->weights; i++)
    {
        if(wit < 5)
            p->data[i] = uRandWeight(-1.f, 1.f) * d; // uniform
        else
            p->data[i] = uRandNormal() * d; // normal
        
        p->momentum[i] = 0.f;
    }

    p->bias = 0.f;
    p->bias_momentum = 0.f;
}

void setWeightInit(network* net, const weight_init_type u)
{
    if(net == NULL){return;}
    net->init = u;
}

void setOptimiser(network* net, const optimiser u)
{
    if(net == NULL){return;}
    net->optimiser = u;
}

void setActivator(network* net, const activator u)
{
    if(net == NULL){return;}
    net->activator = u;
}

void setBatches(network* net, const uint u)
{
    if(net == NULL){return;}
    if(u == 0)
        net->batches = 1;
    else
        net->batches = u;
}

void setLearningRate(network* net, const f32 f)
{
    if(net == NULL){return;}
    net->rate = f;
}

void setGain(network* net, const f32 f)
{
    if(net == NULL){return;}
    net->gain = f;
}

void setUnitDropout(network* net, const f32 f)
{
    if(net == NULL){return;}
    net->dropout = f;
}

void setWeightDropout(network* net, const f32 f)
{
    if(net == NULL){return;}
    net->wdropout = f;
}

void setDropoutDecay(network* net, const f32 f)
{
    if(net == NULL){return;}
    net->dropout_decay = f;
}

void setMomentum(network* net, const f32 f)
{
    if(net == NULL){return;}
    net->momentum = f;
}

void setRMSAlpha(network* net, const f32 f)
{
    if(net == NULL){return;}
    net->rmsalpha = f;
}

void setELUAlpha(network* net, const f32 f)
{
    if(net == NULL){return;}
    net->elualpha = f;
}

void setEpsilon(network* net, const f32 f)
{
    if(net == NULL){return;}
    net->epsilon = f;
}

void randomHyperparameters(network* net)
{
    if(net == NULL){return;}
        
    net->init       = uRand(0, 9);
    net->activator  = uRand(0, 10);
    net->optimiser  = uRand(0, 4);
    net->rate       = uRandFloat(0.001f, 0.1f);
    net->dropout    = uRandFloat(0.f, 0.99f);
    net->wdropout   = uRandFloat(0.f, 0.99f);
    net->momentum   = uRandFloat(0.01f, 0.99f);
    net->rmsalpha   = uRandFloat(0.01f, 0.99f);
    net->elualpha   = uRandFloat(1e-4f, 0.3f);
    net->epsilon    = uRandFloat(1e-8f, 1e-5f);

    net->dropout_decay = uRandFloat(0.f, 0.99f);
    if(net->dropout_decay < 0.1f || net->dropout_decay > 0.9f)
        net->dropout_decay = 0.f;
}

void layerStat(network* net)
{
    f32 min=0.f, avg=0.f, max=0.f;
    
    // layers
    f32 divisor_reciprocal = 1.f/(net->num_layerunits*net->layer[1][0].weights);
    for(int i = 0; i < net->num_layers-1; i++)
    {
        min=0.f, avg=0.f, max=0.f;
        for(int j = 0; j < net->num_layerunits; j++)
        {
            for(uint k = 0; k < net->layer[i][j].weights; k++)
            {
                const f32 w = net->layer[i][j].data[k];
                if(w < min){min = w;}
                else if(w > max){max = w;}
                avg += w;
            }
        }
        printf("%i: %+.3f %+.3f %+.3f [%+.3f]\n", i, min, avg*divisor_reciprocal, max, avg);
    }

    // output layer
    divisor_reciprocal = 1.f/(net->num_outputs*net->layer[net->num_layers-1][0].weights);
    min=0.f, avg=0.f, max=0.f;
    for(int j = 0; j < net->num_outputs; j++)
    {
        for(uint k = 0; k < net->layer[net->num_layers-1][j].weights; k++)
        {
            const f32 w = net->layer[net->num_layers-1][j].data[k];
            if(w < min){min = w;}
            else if(w > max){max = w;}
            avg += w;
        }
    }
    printf("%i: %+.3f %+.3f %+.3f [%+.3f]\n", net->num_layers-1, min, avg*divisor_reciprocal, max, avg);
}

int createNetworkOptimalSmall(network* net, const uint num_inputs, const uint num_outputs)
{
    return createNetwork(net, WEIGHT_INIT_NORMAL_GLOROT, num_inputs, num_outputs, 3, 16, 1);
}

int createNetworkOptimal(network* net, const uint num_inputs, const uint num_outputs)
{
    return createNetwork(net, WEIGHT_INIT_NORMAL_GLOROT, num_inputs, num_outputs, 3, 32, 1);
}

int createNetworkOptimalAccurate(network* net, const uint num_inputs, const uint num_outputs)
{
    return createNetwork(net, WEIGHT_INIT_NORMAL_GLOROT, num_inputs, num_outputs, 4, 32, 1);
}

int createNetwork(network* net, const uint init_weights_type, const uint inputs, const uint num_outputs, const uint hidden_layers, const uint layers_size, const uint default_settings)
{
    const uint layers = hidden_layers+2; //input and output layers

    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(inputs < 1)
        return ERROR_TOOFEWINPUTS;
    if(layers < 3)
        return ERROR_TOOFEWLAYERS;
    if(layers_size < 1)
        return ERROR_TOOSMALL_LAYERSIZE;

    // init net hyper parameters to some default
    net->num_layerunits = layers_size;
    net->num_inputs = inputs;
    net->num_outputs= num_outputs;
    net->num_layers = layers;
    net->init       = init_weights_type;
    if(default_settings == 1)
    {
        net->activator  = TANH;
        net->optimiser  = OPTIM_ADAGRAD;
        net->batches    = 3;
        net->rate       = 0.03f;
        net->gain       = 1.0f;
        net->dropout    = 0.f;
        net->wdropout   = 0.f;
     net->dropout_decay = 0.f;
        net->momentum   = 0.1f;
        net->rmsalpha   = 0.2f;
        net->elualpha   = 0.01f;
        net->epsilon    = 1e-7f;
    }
    net->cbatches = 0;
    
    // create layer output buffers
    net->output = malloc((layers-1) * sizeof(f32*));
    if(net->output == NULL)
    {
        destroyNetwork(net);
        return ERROR_ALLOC_OUTPUT_ARRAY_FAIL;
    }
    for(int i = 0; i < layers-1; i++)
    {
        net->output[i] = malloc(layers_size * sizeof(f32));
        if(net->output[i] == NULL)
        {
            destroyNetwork(net);
            return ERROR_ALLOC_OUTPUT_FAIL;
        }
    }

    // create layers
    net->layer = malloc(layers * sizeof(ptron*));
    if(net->layer == NULL)
    {
        destroyNetwork(net);
        return ERROR_ALLOC_LAYERS_ARRAY_FAIL;
    }
    for(int i = 0; i < layers-1; i++)
    {
        net->layer[i] = malloc(layers_size * sizeof(ptron));
        if(net->layer[i] == NULL)
        {
            destroyNetwork(net);
            return ERROR_ALLOC_LAYERS_FAIL;
        }
    }

    // create output layer
    net->layer[layers-1] = malloc(num_outputs * sizeof(ptron));
    if(net->layer[layers-1] == NULL)
    {
        destroyNetwork(net);
        return ERROR_ALLOC_OUTPUTLAYER_FAIL;
    }

    // create batch output buffer
    net->foutput = malloc(num_outputs * sizeof(f32));
    if(net->foutput == NULL)
    {
        destroyNetwork(net);
        return ERROR_ALLOC_BATCH_OUTPUT_FAIL;
    }

    // create batch error buffer
    net->error = malloc(num_outputs * sizeof(f32));
    if(net->error == NULL)
    {
        destroyNetwork(net);
        return ERROR_ALLOC_BATCH_ERROR_FAIL;
    }

    // init weight
    f32 d = 1.f; //WEIGHT_INIT_UNIFORM / WEIGHT_INIT_NORMAL
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtps(6.0f/(inputs+layers_size));
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrtps(3.0f/inputs);
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN_POW)
        d = powf(inputs, 0.5f);
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_RELU)
        d = sqrtps(6.0f/inputs);
    else if(init_weights_type == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrtps(2.0f/(inputs+layers_size));
    else if(init_weights_type == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrtps(1.0f/inputs);
    else if(init_weights_type == WEIGHT_INIT_NORMAL_RELU)
        d = sqrtps(2.0f/inputs);

    // create first layer perceptrons
    for(int i = 0; i < layers_size; i++)
    {
        if(createPerceptron(&net->layer[0][i], inputs, d, net->init) < 0)
        {
            destroyNetwork(net);
            return ERROR_CREATE_FIRSTLAYER_FAIL;
        }
    }
    
    // weight init
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtps(6.0f/(layers_size+layers_size));
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrtps(3.0f/layers_size);
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_LECUN_POW || init_weights_type == WEIGHT_INIT_NORMAL_LECUN_POW)
        d = powf(layers_size, 0.5f);
    else if(init_weights_type == WEIGHT_INIT_UNIFORM_RELU)
        d = sqrtps(6.0f/layers_size);
    else if(init_weights_type == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrtps(2.0f/(layers_size+layers_size));
    else if(init_weights_type == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrtps(1.0f/layers_size);
    else if(init_weights_type == WEIGHT_INIT_NORMAL_RELU)
        d = sqrtps(2.0f/layers_size);

    // create hidden layers
    for(uint i = 1; i < layers-1; i++)
    {
        for(int j = 0; j < layers_size; j++)
        {
            if(createPerceptron(&net->layer[i][j], layers_size, d, net->init) < 0)
            {
                destroyNetwork(net);
                return ERROR_CREATE_HIDDENLAYER_FAIL;
            }
        }
    }

    // weight init
    if(init_weights_type == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtps(6.0f/(layers_size+1));
    else if(init_weights_type == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrtps(2.0f/(layers_size+1));

    // create output layer
    for(uint i = 0; i < num_outputs; i++)
    {
        if(createPerceptron(&net->layer[layers-1][i], layers_size, d, net->init) < 0)
        {
            destroyNetwork(net);
            return ERROR_CREATE_OUTPUTLAYER_FAIL;
        }
    }

    // memset
    memset(net->foutput, 0x00, net->num_outputs * sizeof(f32));
    memset(net->error, 0x00, net->num_outputs * sizeof(f32));

    // done
    return 0;
}

f32 processNetwork(network* net, const f32* inputs, const f32* target_outputs, f32* outputs)
{
    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(target_outputs == NULL && outputs == NULL)
        return ERROR_BAD_FUNCTION_PARAMETERS;
    
    
/**************************************
    Forward Prop
**************************************/

    // outputs per layer / unit
    f32 of[net->num_layers-1][net->num_layerunits];

    // input layer
    for(int i = 0; i < net->num_layerunits; i++)
        of[0][i] = Activator(doPerceptron(inputs, &net->layer[0][i]), net);

    // hidden layers
    for(int i = 1; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            of[i][j] = Activator(doPerceptron(&of[i-1][0], &net->layer[i][j]), net);

    // output layer
    f32 os[net->num_outputs];
    for(int i = 0; i < net->num_outputs; i++)
        os[i] = doPerceptron(&of[net->num_layers-2][0], &net->layer[net->num_layers-1][0]); // linear

    // if it's just forward pass, return output vector.
    if(target_outputs == NULL && outputs != NULL)
    {
        memcpy(outputs, &os, net->num_outputs * sizeof(f32));
        return 1.f;
    }

/**************************************
    Backward Prop Error (technically "backprop gradient")
**************************************/

    // reset accumulators if cbatches has been reset
    if(net->cbatches == 0)
    {
        for(int i = 0; i < net->num_layers-1; i++)
            memset(net->output[i], 0x00, net->num_layerunits * sizeof(f32));
        
        memset(net->foutput, 0x00, net->num_outputs * sizeof(f32));
        memset(net->error, 0x00, net->num_outputs * sizeof(f32));
    }

    // batch accumulation of outputs    
    for(int i = 0; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            net->output[i][j] += of[i][j];

    // accumulate output error & foutput
    f32 total_loss = 0.f;
    for(int i = 0; i < net->num_outputs; i++)
    {
        net->foutput[i] += os[i];

        //const f32 loss = fabsf(target_outputs[i] - os[i]); // absolute error
        //const f32 loss = powf(target_outputs[i] - os[i], 2.f); // squared error
        const f32 loss = target_outputs[i] - os[i]; // bidirectional error

        net->error[i] += loss;
        total_loss += fabsf(target_outputs[i] - os[i]); // we return actual loss as a metric
    }

    // batching controller
    f32 total_error = 0.f;
    net->cbatches++;
    if(net->cbatches < net->batches)
    {
        if(outputs != NULL)
            memcpy(outputs, &os, net->num_outputs * sizeof(f32));
        return total_loss;
    }
    else
    {
        // divide accumulators to mean
        for(int i = 0; i < net->num_outputs; i++)
        {
            if(isnormal(net->foutput[i]) == 1)
                net->foutput[i] /= net->batches;
            if(isnormal(net->error[i]) == 1)
                net->error[i] /= net->batches; // now becomes mean squared error or mean error for absolute
            total_error += net->error[i];
        }

        for(int i = 0; i < net->num_layers-1; i++)
            for(int j = 0; j < net->num_layerunits; j++)
                if(isnormal(net->output[i][j]) == 1)
                    net->output[i][j] /= net->batches;

        // reset batcher
        net->cbatches = 0;
    }

    // early return if total error is 0 (really unlikely with batching?)
    if(total_error == 0.f)
    {
        if(outputs != NULL)
            memcpy(outputs, &os, net->num_outputs * sizeof(f32));
        return total_loss;
    }

    // define error buffers
    f32 ef[net->num_layers-1][net->num_layerunits];

    // sum output layer error to feed back into hidden layers
    f32 eout = 0.f;
    for(int i = 0; i < net->num_outputs; i++)
        eout += net->gain * net->error[i];

    // output 'derivative error layer' of layer before/behind the output layer
    f32 ler = 0.f;
    for(int j = 0; j < net->layer[net->num_layers-1][0].weights; j++)
        ler += net->layer[net->num_layers-1][0].data[j] * eout;
    ler += net->layer[net->num_layers-1][0].bias * eout;
    for(int i = 0; i < net->num_layerunits; i++)
        ef[net->num_layers-2][i] = net->gain * Derivative(net->output[net->num_layers-2][i], net) * ler;

    // output derivative error of all other layers
    for(int i = net->num_layers-3; i >= 0; i--)
    {
        // compute total error of layer above w.r.t all weights and units of the above layer
        f32 ler = 0.f;
        for(int j = 0; j < net->num_layerunits; j++)
        {
            for(int k = 0; k < net->layer[i+1][j].weights; k++)
                ler += net->layer[i+1][j].data[k] * ef[i+1][j];
            ler += net->layer[i+1][j].bias * ef[i+1][j];
        }
        // propagate that error to into the error variable of each unit of the current layer
        for(int j = 0; j < net->num_layerunits; j++)
            ef[i][j] = net->gain * Derivative(net->output[i][j], net) * ler;
    }

/**************************************
    Update Weights
**************************************/
    
    // update input layer weights
    for(int j = 0; j < net->num_layerunits; j++)
    {
        uint dt = 0;
        if(net->dropout != 0.f && uRandFloat(0.f, 1.f) <= net->dropout)
        {
            if(net->dropout_decay == 0.f)
                continue;
            dt = 1;
        }
        else if(net->wdropout != 0.f)
            dt = 2;
            
        for(int k = 0; k < net->layer[0][j].weights; k++)
            net->layer[0][j].data[k] += doDropout(net, Optimiser(net, inputs[k], ef[0][j], &net->layer[0][j].momentum[k]), dt);

        net->layer[0][j].bias += doDropout(net, Optimiser(net, 1, ef[0][j], &net->layer[0][j].bias_momentum), dt);
    }

    // update hidden layer weights
    for(int i = 1; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            uint dt = 0;
            if(net->dropout != 0.f && uRandFloat(0.f, 1.f) <= net->dropout)
            {
                if(net->dropout_decay == 0.f)
                    continue;
                dt = 1;
            }
            else if(net->wdropout != 0.f)
                dt = 2;
            
            for(int k = 0; k < net->layer[i][j].weights; k++)
                net->layer[i][j].data[k] += doDropout(net, Optimiser(net, net->output[i-1][j], ef[i][j], &net->layer[i][j].momentum[k]), dt);

            net->layer[i][j].bias += doDropout(net, Optimiser(net, 1, ef[i][j], &net->layer[i][j].bias_momentum), dt);
        }
    }

    // update output layer weights
    for(int i = 0; i < net->num_outputs; i++)
    {
        for(int j = 0; j < net->layer[net->num_layers-1][0].weights; j++)
            net->layer[net->num_layers-1][i].data[j] += Optimiser(net, net->output[net->num_layers-2][j], net->error[i], &net->layer[net->num_layers-1][i].momentum[j]);

        net->layer[net->num_layers-1][i].bias += Optimiser(net, 1, net->error[i], &net->layer[net->num_layers-1][i].bias_momentum);
    }

    // done, return forward prop output
    if(outputs != NULL)
        memcpy(outputs, &os, net->num_outputs * sizeof(f32));
    return total_loss;
}

void resetNetwork(network* net)
{
    // validate
    if(net == NULL)
        return;
    if(net->layer == NULL)
        return;

    // reset batching counter
    for(int i = 0; i < net->num_layers-1; i++)
        memset(net->output[i], 0x00, net->num_layerunits * sizeof(f32));
    memset(net->foutput, 0x00, net->num_outputs * sizeof(f32));
    memset(net->error, 0x00, net->num_outputs * sizeof(f32));
    net->cbatches = 0;
    
    // init weight
    f32 d = 1.f; //WEIGHT_INIT_RANDOM
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtps(6.0f/(net->num_inputs+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrtps(3.0f/net->num_inputs);
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN_POW)
        d = powf(net->num_inputs, 0.5f);
    else if(net->init == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrtps(2.0f/(net->num_inputs+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrtps(1.0f/net->num_inputs);

    // reset first layer perceptrons
    for(int i = 0; i < net->num_layerunits; i++)
        resetPerceptron(&net->layer[0][i], d, net->init);
    
    // weight init
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtps(6.0f/(net->num_layerunits+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN)
        d = sqrtps(3.0f/net->num_layerunits);
    else if(net->init == WEIGHT_INIT_UNIFORM_LECUN_POW || net->init == WEIGHT_INIT_NORMAL_LECUN_POW)
        d = powf(net->num_layerunits, 0.5f);
    else if(net->init == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrtps(2.0f/(net->num_layerunits+net->num_layerunits));
    else if(net->init == WEIGHT_INIT_NORMAL_LECUN)
        d = sqrtps(1.0f/net->num_layerunits);

    // reset hidden layers
    for(uint i = 1; i < net->num_layers-1; i++)
        for(int j = 0; j < net->num_layerunits; j++)
            resetPerceptron(&net->layer[i][j], d, net->init);

    // weight init
    if(net->init == WEIGHT_INIT_UNIFORM_GLOROT)
        d = sqrtps(6.0f/(net->num_layerunits+1));
    else if(net->init == WEIGHT_INIT_NORMAL_GLOROT)
        d = sqrtps(2.0f/(net->num_layerunits+1));

    // reset output layer
    resetPerceptron(&net->layer[net->num_layers-1][0], d, net->init);
}

void destroyNetwork(network* net)
{
    // validate
    if(net == NULL)
        return;
    if(net->layer == NULL)
        return;

    // free all perceptron data, percepron units and layers
    for(int i = 0; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            free(net->layer[i][j].data);
            free(net->layer[i][j].momentum);
        }
        free(net->layer[i]);
    }
    free(net->layer[net->num_layers-1][0].data);
    free(net->layer[net->num_layers-1][0].momentum);
    free(net->layer[net->num_layers-1]);
    free(net->layer);
    net->layer = NULL;

    // free output buffers
    for(int i = 0; i < net->num_layers-1; i++)
        free(net->output[i]);
    free(net->output);

    // free batch buffers
    free(net->foutput);
    free(net->error);
}

int saveNetwork(network* net, const char* file)
{
    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    if(net->layer == NULL)
        return ERROR_UNINITIALISED_NETWORK;

    FILE* f = fopen(file, "wb");
    if(f == NULL)
        return -102;

    if(fwrite(&net->num_layerunits, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -103;
    }
#ifdef VERBOSE
    printf("num_layerunits: %u\n", net->num_layerunits);
#endif
    
    if(fwrite(&net->num_inputs, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -104;
    }
#ifdef VERBOSE
    printf("num_inputs: %u\n", net->num_inputs);
#endif

    if(fwrite(&net->num_outputs, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -105;
    }
#ifdef VERBOSE
    printf("num_outputs: %u\n", net->num_outputs);
#endif
    
    if(fwrite(&net->num_layers, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -106;
    }
#ifdef VERBOSE
    printf("num_layers: %u\n", net->num_layers);
#endif
    
    if(fwrite(&net->init, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -107;
    }
#ifdef VERBOSE
    printf("init: %u\n", net->init);
#endif

    if(fwrite(&net->activator, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -108;
    }
#ifdef VERBOSE
    printf("activator: %u\n", net->activator);
#endif

    if(fwrite(&net->optimiser, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -109;
    }
#ifdef VERBOSE
    printf("optimiser: %u\n", net->optimiser);
#endif

    if(fwrite(&net->batches, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -1010;
    }
#ifdef VERBOSE
    printf("batches: %u\n", net->batches);
#endif

    ///

    if(fwrite(&net->rate, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1011;
    }
#ifdef VERBOSE
    printf("rate: %g\n", net->rate);
#endif

    if(fwrite(&net->gain, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1012;
    }
#ifdef VERBOSE
    printf("gain: %g\n", net->gain);
#endif

    if(fwrite(&net->dropout, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1013;
    }
#ifdef VERBOSE
    printf("dropout: %g\n", net->dropout);
#endif

    if(fwrite(&net->wdropout, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1014;
    }
#ifdef VERBOSE
    printf("wdropout: %g\n", net->wdropout);
#endif
    
    if(fwrite(&net->dropout_decay, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1015;
    }
#ifdef VERBOSE
    printf("dropout_decay: %g\n", net->dropout_decay);
#endif

    if(fwrite(&net->momentum, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1016;
    }
#ifdef VERBOSE
    printf("momentum: %g\n", net->momentum);
#endif

    if(fwrite(&net->rmsalpha, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1017;
    }
#ifdef VERBOSE
    printf("rmsalpha: %g\n", net->rmsalpha);
#endif

    if(fwrite(&net->elualpha, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1018;
    }
#ifdef VERBOSE
    printf("elualpha: %g\n", net->elualpha);
#endif
    
    if(fwrite(&net->epsilon, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1019;
    }
#ifdef VERBOSE
    printf("epsilon: %g\n", net->epsilon);
    layerStat(net);
#endif

    ///

    for(int i = 0; i < net->num_layerunits; i++)
    {
        if(fwrite(&net->layer[0][i].data[0], 1, net->num_inputs*sizeof(f32), f) != net->num_inputs*sizeof(f32))
        {
            fclose(f);
            return -1020;
        }
        
        if(fwrite(&net->layer[0][i].momentum[0], 1, net->num_inputs*sizeof(f32), f) != net->num_inputs*sizeof(f32))
        {
            fclose(f);
            return -1021;
        }

        if(fwrite(&net->layer[0][i].bias, 1, sizeof(f32), f) != sizeof(f32))
        {
            fclose(f);
            return -1022;
        }
        
        if(fwrite(&net->layer[0][i].bias_momentum, 1, sizeof(f32), f) != sizeof(f32))
        {
            fclose(f);
            return -1023;
        }
    }

    ///

    for(int i = 1; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            if(fwrite(&net->layer[i][j].data[0], 1, net->num_layerunits*sizeof(f32), f) != net->num_layerunits*sizeof(f32))
            {
                fclose(f);
                return -1024;
            }
            
            if(fwrite(&net->layer[i][j].momentum[0], 1, net->num_layerunits*sizeof(f32), f) != net->num_layerunits*sizeof(f32))
            {
                fclose(f);
                return -1025;
            }

            if(fwrite(&net->layer[i][j].bias, 1, sizeof(f32), f) != sizeof(f32))
            {
                fclose(f);
                return -1026;
            }
            
            if(fwrite(&net->layer[i][j].bias_momentum, 1, sizeof(f32), f) != sizeof(f32))
            {
                fclose(f);
                return -1027;
            }
        }
    }

    ///

    for(int i = 0; i < net->num_outputs; i++)
    {
        if(fwrite(&net->layer[net->num_layers-1][i].data[0], 1, net->num_layerunits*sizeof(f32), f) != net->num_layerunits*sizeof(f32))
        {
            fclose(f);
            return -1028;
        }
        
        if(fwrite(&net->layer[net->num_layers-1][i].momentum[0], 1, net->num_layerunits*sizeof(f32), f) != net->num_layerunits*sizeof(f32))
        {
            fclose(f);
            return -1029;
        }

        if(fwrite(&net->layer[net->num_layers-1][i].bias, 1, sizeof(f32), f) != sizeof(f32))
        {
            fclose(f);
            return -1030;
        }
        
        if(fwrite(&net->layer[net->num_layers-1][i].bias_momentum, 1, sizeof(f32), f) != sizeof(f32))
        {
            fclose(f);
            return -1031;
        }
    }

    ///

    fclose(f);
    return 0;
}

int loadNetwork(network* net, const char* file)
{
    // validate
    if(net == NULL)
        return ERROR_UNINITIALISED_NETWORK;
    
    FILE* f = fopen(file, "rb");
    if(f == NULL)
        return -102;

    ///

    destroyNetwork(net);

    ///

    if(fread(&net->num_layerunits, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -103;
    }
#ifdef VERBOSE
    printf("num_layerunits: %u\n", net->num_layerunits);
#endif

    if(fread(&net->num_inputs, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -104;
    }
#ifdef VERBOSE
    printf("num_inputs: %u\n", net->num_inputs);
#endif

    if(fread(&net->num_outputs, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -105;
    }
#ifdef VERBOSE
    printf("num_outputs: %u\n", net->num_outputs);
#endif

    if(fread(&net->num_layers, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -106;
    }
#ifdef VERBOSE
    printf("num_layers: %u\n", net->num_layers);
#endif

    if(fread(&net->init, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -107;
    }
#ifdef VERBOSE
    printf("init: %u\n", net->init);
#endif

    if(fread(&net->activator, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -108;
    }
#ifdef VERBOSE
    printf("activator: %u\n", net->activator);
#endif

    if(fread(&net->optimiser, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -109;
    }
#ifdef VERBOSE
    printf("optimiser: %u\n", net->optimiser);
#endif

    if(fread(&net->batches, 1, sizeof(uint), f) != sizeof(uint))
    {
        fclose(f);
        return -1010;
    }
#ifdef VERBOSE
    printf("batches: %u\n", net->batches);
#endif

    ///

    if(fread(&net->rate, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1011;
    }
#ifdef VERBOSE
    printf("rate: %g\n", net->rate);
#endif

    if(fread(&net->gain, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1012;
    }
#ifdef VERBOSE
    printf("gain: %g\n", net->gain);
#endif

    if(fread(&net->dropout, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1013;
    }
#ifdef VERBOSE
    printf("dropout: %g\n", net->dropout);
#endif

    if(fread(&net->wdropout, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1014;
    }
#ifdef VERBOSE
    printf("wdropout: %g\n", net->wdropout);
#endif

    if(fread(&net->dropout_decay, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1015;
    }
#ifdef VERBOSE
    printf("dropout_decay: %g\n", net->dropout_decay);
#endif

    if(fread(&net->momentum, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1016;
    }
#ifdef VERBOSE
    printf("momentum: %g\n", net->momentum);
#endif

    if(fread(&net->rmsalpha, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1017;
    }
#ifdef VERBOSE
    printf("rmsalpha: %g\n", net->rmsalpha);
#endif

    if(fread(&net->elualpha, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1018;
    }
#ifdef VERBOSE
    printf("elualpha: %g\n", net->elualpha);
#endif
    
    if(fread(&net->epsilon, 1, sizeof(f32), f) != sizeof(f32))
    {
        fclose(f);
        return -1019;
    }
#ifdef VERBOSE
    printf("epsilon: %g\n", net->epsilon);
#endif

    ///

    createNetwork(net, net->init, net->num_inputs, net->num_outputs, net->num_layers-2, net->num_layerunits, 0);

    ///

    for(int i = 0; i < net->num_layerunits; i++)
    {
        if(fread(&net->layer[0][i].data[0], 1, net->num_inputs*sizeof(f32), f) != net->num_inputs*sizeof(f32))
        {
            fclose(f);
            return -1020;
        }

        if(fread(&net->layer[0][i].momentum[0], 1, net->num_inputs*sizeof(f32), f) != net->num_inputs*sizeof(f32))
        {
            fclose(f);
            return -1021;
        }

        if(fread(&net->layer[0][i].bias, 1, sizeof(f32), f) != sizeof(f32))
        {
            fclose(f);
            return -1022;
        }

        if(fread(&net->layer[0][i].bias_momentum, 1, sizeof(f32), f) != sizeof(f32))
        {
            fclose(f);
            return -1023;
        }
    }

    ///

    for(int i = 1; i < net->num_layers-1; i++)
    {
        for(int j = 0; j < net->num_layerunits; j++)
        {
            if(fread(&net->layer[i][j].data[0], 1, net->num_layerunits*sizeof(f32), f) != net->num_layerunits*sizeof(f32))
            {
                fclose(f);
                return -1024;
            }

            if(fread(&net->layer[i][j].momentum[0], 1, net->num_layerunits*sizeof(f32), f) != net->num_layerunits*sizeof(f32))
            {
                fclose(f);
                return -1025;
            }

            if(fread(&net->layer[i][j].bias, 1, sizeof(f32), f) != sizeof(f32))
            {
                fclose(f);
                return -1026;
            }

            if(fread(&net->layer[i][j].bias_momentum, 1, sizeof(f32), f) != sizeof(f32))
            {
                fclose(f);
                return -1027;
            }
        }
    }

    ///

    for(int i = 0; i < net->num_outputs; i++)
    {
        if(fread(&net->layer[net->num_layers-1][i].data[0], 1, net->num_layerunits*sizeof(f32), f) != net->num_layerunits*sizeof(f32))
        {
            fclose(f);
            return -1028;
        }

        if(fread(&net->layer[net->num_layers-1][i].momentum[0], 1, net->num_layerunits*sizeof(f32), f) != net->num_layerunits*sizeof(f32))
        {
            fclose(f);
            return -1029;
        }

        if(fread(&net->layer[net->num_layers-1][i].bias, 1, sizeof(f32), f) != sizeof(f32))
        {
            fclose(f);
            return -1030;
        }

        if(fread(&net->layer[net->num_layers-1][i].bias_momentum, 1, sizeof(f32), f) != sizeof(f32))
        {
            fclose(f);
            return -1031;
        }
    }

    ///
    
    fclose(f);
#ifdef VERBOSE
    layerStat(net);
#endif
    return 0;
}

#endif
