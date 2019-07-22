#include "ann.h"

const char *ann_status_str(ann_status_t status) {

#define CASE_ANN_STATUS(status)                                                \
  case ANN_STATUS_##status:                                                    \
    ptr = #status;                                                             \
    break

  char const *ptr = NULL;
  switch (status) {
    CASE_ANN_STATUS(OK);
    CASE_ANN_STATUS(ARG_NULL);
    CASE_ANN_STATUS(ARG_ERR);
    CASE_ANN_STATUS(INPUT_LAYER_EXISTS);
    CASE_ANN_STATUS(INPUT_LAYER_MISSING);
    CASE_ANN_STATUS(OUTPUT_LAYER_EXISTS);
    CASE_ANN_STATUS(OUTPUT_LAYER_MISSING);
    CASE_ANN_STATUS(NEURON_TYPE_ERR);
  default:
    ptr = "UNKOWN";
    break;
  }
  return ptr;
#undef CASE_ANN_STATUS
}

const char *ann_neuron_type_str(ann_neuron_type_t n_type) {

#define CASE_ANN_NEURON(type)                                                  \
  case ANN_NEURON_TYPE_##type:                                                 \
    ptr = #type;                                                               \
    break

  char const *ptr = NULL;
  switch (n_type) {
    CASE_ANN_NEURON(SIGMOID);
    CASE_ANN_NEURON(TANH);
    CASE_ANN_NEURON(RELU);
    CASE_ANN_NEURON(LEAKY_RELU);
    CASE_ANN_NEURON(LINEAR);
  default:
    ptr = "UNKOWN";
    break;
  }
  return ptr;
#undef CASE_ANN_NEURON
}

extern int random_int(void);

void random_fp(fp_t *fp, size_t len) {
  for (size_t i = 0; i < len; i++) {
    fp[i] = FP_SDIV(random_int(), FP(INT_MAX));
  }
}

#include <stdio.h>

void ann_print_neuron(ann_neuron_t *neuron, size_t w_length) {

  fprintf(stdout, "    {\n");
  fprintf(stdout, "     \"ntype\": \"%s\",\n",
          ann_neuron_type_str(neuron->ntype));

  fprintf(stdout, "     \"weights\": [\n");
  if (w_length) {
    fprintf(stdout, "      %.12f,\n", neuron->weights[0]);
    for (size_t i = 1; i < w_length + 1; i++) {
      fprintf(stdout, "      %.12f", neuron->weights[i]);
      fprintf(stdout, (i == (w_length)) ? "\n" : ",\n");
    }
  }

  fprintf(stdout, "     ],\n");

  fprintf(stdout, "     \"output\": %.12f\n", neuron->output);

  fprintf(stdout, "    }");
}

void ann_print_layer(ann_layer_t *layer, size_t w_length) {
  fprintf(stdout, "  {\n");
  fprintf(stdout, "   \"neurons\": [\n");
  for (size_t i = 0; i < layer->length; i++) {
    ann_print_neuron(&layer->neurons[i], w_length);
    fprintf(stdout, (i == (layer->length - 1)) ? "\n" : ",\n");
  }
  fprintf(stdout, "   ]\n");
  fprintf(stdout, "  }");
}

void ann_print_ann(ann_t *ann) {

  ann_layer_t *layer = ann->input;

  fprintf(stdout, "{\n");
  fprintf(stdout, " \"layers\": [\n");
  while (layer != NULL) {
    size_t w_length = layer->prev ? layer->prev->length : 0;

    ann_print_layer(layer, w_length);

    fprintf(stdout, (layer->next == NULL) ? "\n" : ",\n");

    layer = layer->next;
  }
  fprintf(stdout, " ]\n");
  fprintf(stdout, "}\n");
}

bool ann_predict(fp_t output, fp_t threshold) { return output >= threshold; }

void ann_map_feautre(fp_t x0, fp_t x1, fp_t *map, size_t order) {
  // Feature mapping function to polynomial features
  size_t term = 0;
  for (size_t i = 0; i <= order; i++) {
    for (size_t j = 0; j <= i; j++) {
      map[term++] = FP_POW(x0, i - j) * FP_POW(x1, j);
    }
  }
}

static fp_t ann_activate_sigmoid(fp_t sum) {
  /* 1 / 1 (e ^-x) */
  fp_t g;

  g = FP_SMUL(sum, FP(-1));
  g = FP_EXP(g);
  g = FP_SADD(FP(1), g);

  g = FP_SDIV(FP(1), g);

  return g;
}

static fp_t ann_activate_tanh(fp_t sum) {
  /* 2*sigmoid(2*x) - 1 */
  fp_t g;

  g = FP_SMUL(sum, FP(2));
  g = ann_activate_sigmoid(g);
  g = FP_SMUL(g, FP(2));
  g = FP_SSUB(g, FP(1));
  return g;
}

static fp_t ann_activate_param_relu(fp_t a, fp_t sum) {
  /* max(x * a, x) */

  fp_t a_sum = FP_SMUL(a, sum);

  return FP_MAX(a_sum, sum);
}

static fp_t ann_activate_leaky_relu(fp_t sum) {
  return ann_activate_param_relu(FP(0.01), sum);
}

static fp_t ann_activate_relu(fp_t sum) {
  /* max(0, x) */
  return ann_activate_param_relu(FP(0), sum);
}

static fp_t ann_activate_linear(fp_t a, fp_t sum) { return FP_SMUL(a, sum); }

static ann_layer_t *ann_layer_last_get(ann_t *ann) {
  ann_layer_t *last_layer = ann->input;

  while (last_layer != NULL && last_layer->next != NULL) {
    last_layer = last_layer->next;
  }

  return last_layer;
}

#include <assert.h>

ann_status_t ann_set_layer_type(ann_layer_t *l, ann_neuron_type_t ntype) {

  if (l == NULL) {
    return ANN_STATUS_ARG_NULL;
  }
  if (ntype >= ANN_NEURON_TYPE_NUM) {
    return ANN_STATUS_ARG_ERR;
  }

  for (size_t i = 0; i < l->length; i++) {
    l->neurons[i].ntype = ntype;
  }

  return ANN_STATUS_OK;
}

ann_status_t ann_add_layer_input(ann_t *ann, ann_layer_t *layer) {

  if (ann == NULL || layer == NULL) {
    return ANN_STATUS_ARG_NULL;
  }

  if (ann->input != NULL) {
    return ANN_STATUS_INPUT_LAYER_EXISTS;
  }

  ann->input = layer;

  layer->next = ann->output;

  layer->prev = NULL;

  return ANN_STATUS_OK;
}

ann_status_t ann_add_layer_hidden(ann_t *ann, ann_layer_t *layer) {

  if (ann == NULL || layer == NULL) {
    return ANN_STATUS_ARG_NULL;
  }

  if (ann->input == NULL) {
    return ANN_STATUS_INPUT_LAYER_MISSING;
  }

  ann_layer_t *last_layer = ann_layer_last_get(ann);

  /* If the last layer is the output layer, put this guy in
   * between, otherwise, just stick this guy at the end and doubly
   * link the list */
  if (last_layer == ann->output) {
    layer->next = last_layer;
    layer->prev = last_layer->prev;
    layer->prev->next = layer;
    last_layer->prev = layer;
  } else {
    last_layer->next = layer;
    layer->prev = last_layer;
  }

  return ANN_STATUS_OK;
}

ann_status_t ann_add_layer_output(ann_t *ann, ann_layer_t *layer) {
  if (ann == NULL || layer == NULL) {
    return ANN_STATUS_ARG_NULL;
  }

  if (ann->output != NULL) {
    return ANN_STATUS_OUTPUT_LAYER_EXISTS;
  }

  ann->output = layer;

  ann->output->prev = ann_layer_last_get(ann);
  ann->output->prev->next = layer;

  ann->output->next = NULL;

  return ANN_STATUS_OK;
}

static ann_status_t ann_activate_layer(ann_layer_t *layer) {
  if (layer == NULL) {
    return ANN_STATUS_ARG_NULL;
  }

  for (size_t i = 0; i < layer->length; i++) {

    fp_t outout = 0;
    ann_neuron_t *neuron = &layer->neurons[i];

    /* Init the work neuron to bias */
    neuron->output = neuron->weights[0];

    const fp_t *weights = &neuron->weights[1];
    /* Assign the sum of products of the inputs * weights to the
     * neuron's output */
    for (size_t j = 0; j < layer->prev->length; j++) {

      outout = FP_SMUL(weights[j], layer->prev->neurons[j].output);

      neuron->output = FP_SADD(neuron->output, outout);
    }

    /* Fire the correct activation function for the neuron's type */
    switch (neuron->ntype) {
    case ANN_NEURON_TYPE_SIGMOID:
      neuron->output = ann_activate_sigmoid(neuron->output);
      break;
    case ANN_NEURON_TYPE_TANH:
      neuron->output = ann_activate_tanh(neuron->output);
      break;
    case ANN_NEURON_TYPE_RELU:
      neuron->output = ann_activate_relu(neuron->output);
      break;
    case ANN_NEURON_TYPE_LEAKY_RELU:
      neuron->output = ann_activate_leaky_relu(neuron->output);
      break;
    case ANN_NEURON_TYPE_LINEAR:
      neuron->output = ann_activate_linear(FP(1), neuron->output);
      break;
    default:
      return ANN_STATUS_NEURON_TYPE_ERR;
    }
  }

  return ANN_STATUS_OK;
}

ann_status_t ann_activate(ann_t *ann, const fp_t *inputs, fp_t *outputs) {
  if (ann == NULL || inputs == NULL) {
    return ANN_STATUS_ARG_NULL;
  }

  if (ann->input == NULL) {
    return ANN_STATUS_INPUT_LAYER_MISSING;
  }

  if (ann->output == NULL) {
    return ANN_STATUS_OUTPUT_LAYER_MISSING;
  }

  /* Move the inputs to the ouput of the input layer (input layer
   * applies no bias or weight on its own, so it's just a
   * passthrough) */
  for (size_t i = 0; i < ann->input->length; i++) {
    ann->input->neurons[i].output = inputs[i];
  }

  /* Activate each layer in turn. Continue until the output layer is
   * reached, then copy the final layer's outputs to the output
   * holding buffer (assuming non-null) */
  ann_layer_t *layer = ann->input->next;

  while (layer != NULL) {
    ann_status_t status = ann_activate_layer(layer);

    if (status != ANN_STATUS_OK) {
      return status;
    }
    layer = layer->next;
  }

  if (outputs != NULL) {
    for (size_t i = 0; i < ann->output->length; i++) {
      outputs[i] = ann->output->neurons[i].output;
    }
  }

  return ANN_STATUS_OK;
}
