#ifndef _ANN_H
#define _ANN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

typedef double fp_t;

#include <limits.h>
#include <math.h>

#define FP(n) (1.0 * n)

#define FP_SMUL(a, b) (a * b)
#define FP_SDIV(a, b) (a / b)

#define FP_EXP(a) exp(a)

#define FP_SADD(a, b) (a + b)
#define FP_SSUB(a, b) (a - b)

#define FP_POW(a, b) pow(a, b)
#define FP_MAX(a, b) fmax(a, b)
#define FP_ABS(a) fabs(a)

typedef enum ann_status_e {
  ANN_STATUS_OK = 0,
  ANN_STATUS_ARG_NULL,
  ANN_STATUS_ARG_ERR,
  ANN_STATUS_INPUT_LAYER_EXISTS,
  ANN_STATUS_INPUT_LAYER_MISSING,
  ANN_STATUS_OUTPUT_LAYER_EXISTS,
  ANN_STATUS_OUTPUT_LAYER_MISSING,
  ANN_STATUS_NEURON_TYPE_ERR,

  ANN_STATUS_NUM
} ann_status_t;

typedef enum ann_neuron_type_e {
  ANN_NEURON_TYPE_LINEAR = 0,
  ANN_NEURON_TYPE_SIGMOID,
  ANN_NEURON_TYPE_TANH,
  ANN_NEURON_TYPE_RELU,
  ANN_NEURON_TYPE_LEAKY_RELU,

  ANN_NEURON_TYPE_NUM
} ann_neuron_type_t;

typedef struct __attribute__((__packed__)) ann_neuron_s {
  ann_neuron_type_t ntype;
  const fp_t *weights;
  fp_t output;
} ann_neuron_t;

typedef struct ann_layer_s {
  size_t length;
  struct ann_layer_s *prev;
  struct ann_layer_s *next;
  ann_neuron_t *neurons;
} ann_layer_t;

typedef struct ann_s {
  size_t length;
  ann_layer_t *input;
  ann_layer_t *output;
} ann_t;

#define _DECLARE_ANN_LAYER(name, len, statik)                                  \
  statik ann_neuron_t name##_neurons[len] = {0};                               \
  statik ann_layer_t name = {.neurons = name##_neurons, .length = len};

#define DECLARE_ANN_LAYER(name, len) _DECLARE_ANN_LAYER(name, len, )
#define DECLARE_STATIC_ANN_LAYER(name, len)                                    \
  _DECLARE_ANN_LAYER(name, len, static)

void random_fp(fp_t *fp, size_t len);

#include <stdbool.h>

bool ann_predict(fp_t output, fp_t threshold);

void ann_map_feautre(fp_t x0, fp_t x1, fp_t *map, size_t order);

void ann_print_ann(ann_t *ann);

const char *ann_neuron_type_str(ann_neuron_type_t n_type);

ann_status_t ann_set_layer_type(ann_layer_t *l, ann_neuron_type_t ntype);

ann_status_t ann_add_layer_input(ann_t *ann, ann_layer_t *layer);

ann_status_t ann_add_layer_hidden(ann_t *ann, ann_layer_t *layer);

ann_status_t ann_add_layer_output(ann_t *ann, ann_layer_t *layer);

ann_status_t ann_activate(ann_t *ann, const fp_t *inputs, fp_t *outputs);

#ifdef __cplusplus
}
#endif

#endif /* _ANN_H */