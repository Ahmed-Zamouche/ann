#include "ann.h"

#include <stdlib.h>
#include <time.h>

extern int TEST_ann(int argc, char const *argv[]);
extern int TEST_linear_reg(int argc, char const *argv[]);
extern int TEST_logistic_reg(int argc, char const *argv[]);
extern int TEST_logistic_reg_one_vs_all(int argc, char const *argv[]);

int random_int(void) {
  static int init = 0;
  if (!init) {
    srand(time(NULL));
    init = 1;
  }
  return rand();
}

void random_bytes(uint8_t *buf, size_t len) {
  srand(time(NULL)); // Initialization, should only be called once.
  for (size_t i = 0; i < len; i++) {
    buf[i] = rand();
  }
}

int main(int argc, char const *argv[]) {
  // TEST_ann(argc, argv);
  TEST_linear_reg(argc, argv);
  TEST_logistic_reg(argc, argv);
  TEST_logistic_reg_one_vs_all(argc, argv);
}