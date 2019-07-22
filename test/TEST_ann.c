
#include "ann.h"

#include <stdlib.h>
#include <time.h>

DECLARE_STATIC_ANN_LAYER(layer_input, 2);

DECLARE_STATIC_ANN_LAYER(layer_hidden, 3);
static fp_t weights_hidden[3][1 + 2];

DECLARE_STATIC_ANN_LAYER(layer_output, 2);
static fp_t weights_output[2][1 + 3];

static ann_t ann_obj = {0};

int TEST_ann(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;

  // fprintf(stdout, "TEST ANN\n");

  ann_add_layer_input(&ann_obj, &layer_input);
  ann_add_layer_hidden(&ann_obj, &layer_hidden);
  random_fp(weights_hidden[0], sizeof(weights_hidden) / sizeof(fp_t));
  for (size_t i = 0; i < layer_hidden.length; i++) {
    layer_hidden.neurons[i].weights = weights_hidden[i];
  }

  ann_add_layer_output(&ann_obj, &layer_output);
  random_fp(weights_output[0], sizeof(weights_output) / sizeof(fp_t));
  for (size_t i = 0; i < layer_output.length; i++) {
    layer_output.neurons[i].weights = weights_output[i];
  }

  ann_print_ann(&ann_obj);

  return 0;
}