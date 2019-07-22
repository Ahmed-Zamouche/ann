#include "ann.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

DECLARE_STATIC_ANN_LAYER(layer_input, 1);
DECLARE_STATIC_ANN_LAYER(layer_output, 1);
static const fp_t weights_output[1][1 + 1];
static ann_t ann_obj = {0};

static const fp_t weights_output[1][1 + 1] = {{FP(-3.6303), FP(1.1664)}};

DECLARE_STATIC_ANN_LAYER(layer_input_multi, 2);
DECLARE_STATIC_ANN_LAYER(layer_output_multi, 1);
static const fp_t weights_output_multi[1][2 + 1];
static ann_t ann_obj_multi = {0};

static const fp_t weights_output_multi[1][2 + 1] = {
    {FP(340397.96354), FP(109848.00846), FP(-5866.45408)}};

int TEST_linear_reg_multi(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;

  fprintf(stdout, "Test linear regression multi features.\n");

  ann_add_layer_input(&ann_obj_multi, &layer_input_multi);

  ann_add_layer_output(&ann_obj_multi, &layer_output_multi);

  for (size_t i = 0; i < layer_output_multi.length; i++) {
    layer_output_multi.neurons[i].weights = weights_output_multi[i];
  }

  // Input are normalized using mu and sigma
  fp_t mu[] = {FP(2000.6809), FP(3.1702)};
  fp_t sigma[] = {FP(794.70235), FP(0.76098)};

  // predict [1650 , 3] -> 293237.16148
  fp_t input[2] = {FP_SDIV(FP_SSUB(FP(1650), mu[0]), sigma[0]),
                   FP_SDIV(FP_SSUB(FP(3), mu[1]), sigma[1])};
  fp_t outputs;

  ann_activate(&ann_obj_multi, input, &outputs);

  // ann_print_ann(&ann_obj_multi);

  assert(FP_ABS(outputs - FP(293237.16148)) < FP(0.5));

  return 0;
}
int TEST_linear_reg_single(int argc, char const *argv[]) {
  (void)argc;
  (void)argv;

  fprintf(stdout, "Test linear regression single feature.\n");

  ann_add_layer_input(&ann_obj, &layer_input);

  ann_add_layer_output(&ann_obj, &layer_output);

  for (size_t i = 0; i < layer_output.length; i++) {
    layer_output.neurons[i].weights = weights_output[i];
  }

  fp_t input, outputs;
  // predict 1.8 -> -1.530780
  input = FP(1.8);
  ann_activate(&ann_obj, &input, &outputs);
  assert(FP_ABS(outputs - FP(-1.530780)) < FP(1E-3));

  // predict 1.8 -> -0.71439
  input = FP(2.5);
  ann_activate(&ann_obj, &input, &outputs);

  // ann_print_ann(&ann_obj);

  assert(FP_ABS(outputs - FP(-0.71439)) < FP(1E-3));

  return 0;
}

int TEST_linear_reg(int argc, char const *argv[]) {
  TEST_linear_reg_single(argc, argv);
  TEST_linear_reg_multi(argc, argv);
  return 0;
}