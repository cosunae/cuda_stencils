#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

#include "../defs.h"
#include "libjson.h"
#include "mesh.hpp"

int parse_uint(const char *str, unsigned int *output) {
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

int main(int argc, char **argv) {

  unsigned int isize = 512;
  unsigned int jsize = 512;
  unsigned int ksize = 60;

  for (int i = 1; i < argc; i++) {
    if (!std::string("--isize").compare(argv[i])) {
      if (++i >= argc || !parse_uint(argv[i], &isize)) {
        std::cerr << "Wrong parsing" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    if (!std::string("--jsize").compare(argv[i])) {
      if (++i >= argc || !parse_uint(argv[i], &jsize)) {
        std::cerr << "Wrong parsing" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    if (!std::string("--ksize").compare(argv[i])) {
      if (++i >= argc || !parse_uint(argv[i], &ksize)) {
        std::cerr << "Wrong parsing" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  printf("Configuration :\n");
  printf("  isize: %d\n", isize);
  printf("  jsize: %d\n", jsize);
  printf("  ksize: %d\n", ksize);

  const size_t tot_size = isize * jsize * ksize;
  const size_t tsteps = 100;
  const size_t warmup_step = 10;

  mesh mesh_(isize, jsize, 2);
  mesh_.test();
  mesh_.print();
}
