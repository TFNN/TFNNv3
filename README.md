## TFCNNv3 - Tiny Fully Connected Neural Network Library
The example project provided is based on work by [Jim C. Williams](https://github.com/jcwml/neural_zodiac).

### Update Log
`[05/08/22]` - Initial commit, includes working example training on the Zodiac compatibility dataset.<br>
`[06/08/22]` - A [HOGWILD!](https://arxiv.org/pdf/1106.5730.pdf) implementation has been added to the [/hogwild](/hogwild) directory.<br>
`[10/08/22]` - Added support for SSE inverse square root `rsqrtss` in ADAGRAD and RMSPROP.<br>
`[11/08/22]` - Fixed `saveNetwork()` & `loadNetwork()` functions & added three new `createNetwork()` functions.<br>

### Notes
- Pass `target_outputs` as NULL to `processNetwork()` for a forward pass / no training.
- Turning off `FAST_PREDICTABLE_MODE` will use the platform dependent (Linux/Unix) `/dev/urandom`, it's two times slower but has higher entropy.
- Turning off `NOSSE` will enable the use of SSE for square roots, which actually can be slower.
- **Why no softmax?**<br>Softmax is targeted more towards classification of categories, which is better in a CNN where your outputs are onehot vector category classes. Here we have linear output layers because they fit well to a wider range of applications.
- **Why no ADAM optimiser?**<br>Requires an extra parameter per weight, too much memory bandwidth usage over ADAGRAD.

### Functionality overview
```
// simplified createNetwork() functions
int  createNetworkOptimalSmall(network* net, const uint num_inputs, const uint num_outputs);    // 8.2 KiB (8,396 bytes)
int  createNetworkOptimal(network* net, const uint num_inputs, const uint num_outputs);         // 28.3 KiB (29,004 bytes)
int  createNetworkOptimalAccurate(network* net, const uint num_inputs, const uint num_outputs); // 36.6 KiB (37,452 bytes)
/*
    createNetworkOptimalSmall()     - Smallest network for reasonable results.
    createNetworkOptimalAccurate()  - Largest network for more accurate results.
    createNetworkOptimal()          - The best of both worlds, the default for most networks.
*/

// primary function set
int  createNetwork(network* net, const weight_init_type init_type, const uint num_inputs, const uint num_outputs, const uint num_hidden_layers, const uint num_layer_units, const uint default_settings);
f32  processNetwork(network* net, const f32* inputs, const f32* target_outputs, f32* outputs);
void resetNetwork(network* net);
void destroyNetwork(network* net);
int  saveNetwork(network* net, const char* file);
int  loadNetwork(network* net, const char* file);

// debugging
void layerStat(network* net);
/*
    This is a method of getting a concise overview of network weights
    per layer in the form; layer: min avg max [sum]
    
    That's the min, average, and max weight in the specific layer
    followed by the summation of all weights in the layer in
    squared brackets.
    
    Layer 0 is the input layer, 1-x are hidden layers and the final
    layer is the output.
    
    That is enough to give you a good idea of how the weights are
    scaling per layer during the training process.
*/

// accessors
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

// random functions
f32  uRandNormal();
f32  uRandFloat(const f32 min, const f32 max);
f32  uRandWeight(const f32 min, const f32 max);
uint uRand(const uint min, const uint umax);
void srandf(const int seed);

// enums
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
```
