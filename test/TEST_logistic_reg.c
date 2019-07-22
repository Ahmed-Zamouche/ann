
#include "ann.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

DECLARE_STATIC_ANN_LAYER(layer_input, 2);
DECLARE_STATIC_ANN_LAYER(layer_output, 1);
static const fp_t weights_output[1][2 + 1];
static ann_t ann_obj = {0};

static const fp_t weights_output[1][2 + 1] = {
    {FP(-25.16127), FP(0.20623), FP(0.201470)}};

DECLARE_STATIC_ANN_LAYER(layer_input_reg, 27);
DECLARE_STATIC_ANN_LAYER(layer_output_reg, 1);
static const fp_t weights_output_reg[1][27 + 1];
static ann_t ann_obj_reg = {0};

static const fp_t weights_output_reg[1][27 + 1] = {
    {FP(1.273005),  FP(0.624876),  FP(1.177376),  FP(-2.020142), FP(-0.912616),
     FP(-1.429907), FP(0.125668),  FP(-0.368551), FP(-0.360033), FP(-0.171068),
     FP(-1.460894), FP(-0.052499), FP(-0.618889), FP(-0.273745), FP(-1.192301),
     FP(-0.240993), FP(-0.207934), FP(-0.047224), FP(-0.278327), FP(-0.296602),
     FP(-0.453957), FP(-1.045511), FP(0.026463),  FP(-0.294330), FP(0.014381),
     FP(-0.328703), FP(-0.143796), FP(-0.924883)}};

int TEST_logistic_regression_reg(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;

  fprintf(stdout, "Test logistic regression with regularization.\n");

  ann_set_layer_type(&layer_input_reg, ANN_NEURON_TYPE_SIGMOID);
  ann_add_layer_input(&ann_obj_reg, &layer_input_reg);

  ann_set_layer_type(&layer_output_reg, ANN_NEURON_TYPE_SIGMOID);
  ann_add_layer_output(&ann_obj_reg, &layer_output_reg);

  for (size_t i = 0; i < layer_output_reg.length; i++) {
    layer_output_reg.neurons[i].weights = weights_output_reg[i];
  }
  // predict [ 1.0318e-08,   1.2819e-07,   1.5926e-06,   1.9787e-05,
  // 2.4582e-04,   3.0541e-03,   3.7943e-02] -> 0.776289
  fp_t input[27 + 1] = {0};
  fp_t outputs;

  ann_map_feautre(FP(0.051267), FP(0.69956), input, 6);

  ann_activate(&ann_obj_reg, &input[1], &outputs);

  // ann_print_ann(&ann_obj_reg);

  assert(FP_ABS(outputs - FP(0.698812)) < FP(1E-3));

  return 0;
}

int TEST_logistic_regression(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;

  fprintf(stdout, "Test logistic regression features.\n");

  ann_set_layer_type(&layer_input, ANN_NEURON_TYPE_SIGMOID);
  ann_add_layer_input(&ann_obj, &layer_input);

  ann_set_layer_type(&layer_output, ANN_NEURON_TYPE_SIGMOID);
  ann_add_layer_output(&ann_obj, &layer_output);

  for (size_t i = 0; i < layer_output.length; i++) {
    layer_output.neurons[i].weights = weights_output[i];
  }
  // predict [45 , 85] -> 0.776289
  fp_t input[2] = {FP(45), FP(85)};
  fp_t outputs;

  ann_activate(&ann_obj, input, &outputs);

  // ann_print_ann(&ann_obj);

  assert(FP_ABS(outputs - FP(0.776289)) < FP(1E-3));

  return 0;
}

int TEST_logistic_reg(int argc, char const *argv[]) {
  TEST_logistic_regression(argc, argv);
  TEST_logistic_regression_reg(argc, argv);
  return 0;
}
