#include <iostream>
#include <vector>
#include <chrono>

#include "defs.h"

#define FNNAME(a) a
#define LOAD(a) __ldg(& a)

#include "stencil_kernels.h"

#undef FNNAME
#undef LOAD


#define FNNAME(a) a##_ldg
#define LOAD(a) a

#include "stencil_kernels.h"

#undef FNNAME
#undef LOAD

int parse_uint(const char *str, unsigned int *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}


int main(int argc, char** argv) {

    unsigned int isize=512;
    unsigned int jsize=512;
    unsigned int ksize=60;

    for (int i = 1; i < argc; i++)
    {
        if (!std::string("--isize").compare(argv[i])) {
            if (++i >= argc || !parse_uint(argv[i], &isize))
            {
                std::cerr << "Wrong parsing" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        if (!std::string("--jsize").compare(argv[i])) {
            if (++i >= argc || !parse_uint(argv[i], &jsize))
            {
                std::cerr << "Wrong parsing" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        if (!std::string("--ksize").compare(argv[i])) {
            if (++i >= argc || !parse_uint(argv[i], &ksize))
            {
                std::cerr << "Wrong parsing" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    printf("Configuration :\n");
    printf("  isize: %d\n",isize);
    printf("  jsize: %d\n",jsize);
    printf("  ksize: %d\n",ksize);

    const size_t tot_size = isize*jsize*ksize;
    const size_t tsteps=100;
    const size_t warmup_step=10;
 
    printf("======================== FLOAT =======================\n");
    std::vector<double> timings(num_bench_st);
    std::fill(timings.begin(),timings.end(), 0);

    launch<float>(timings, isize, jsize, ksize, tsteps, warmup_step);

    printf("-------------    NO TEXTURE   -------------\n");
    printf("copy : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[copy_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[copy_st]);
    printf("copyi1 : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[copyi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[copyi1_st]);
    printf("sumi1 : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[sumi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumi1_st]);
    printf("sumj1 : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[sumj1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumj1_st]);
    printf("sumk1 : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[sumk1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumk1_st]);
    printf("lap : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[lap_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[lap_st]);

    std::fill(timings.begin(),timings.end(), 0);
    launch_ldg<float>(timings, isize, jsize, ksize, tsteps, warmup_step);

    printf("-------------   TEXTURE   -------------\n");
    printf("copy : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[copy_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[copy_st]);
    printf("copyi1 : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[copyi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[copyi1_st]);
    printf("sumi1 : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[sumi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumi1_st]);
    printf("sumj1 : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[sumj1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumj1_st]);
    printf("sumk1 : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[sumk1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumk1_st]);
    printf("lap : %f GB/s, time : %f \n", tot_size*2*sizeof(float)/(timings[lap_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[lap_st]);

    printf("======================== DOUBLE =======================\n");

    std::fill(timings.begin(),timings.end(), 0);
    launch<double>(timings, isize, jsize, ksize, tsteps, warmup_step);

    printf("-------------    NO TEXTURE   -------------\n");
    printf("copy : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[copy_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[copy_st]);
    printf("copyi1 : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[copyi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[copyi1_st]);
    printf("sumi1 : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[sumi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumi1_st]);
    printf("sumj1 : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[sumj1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumj1_st]);
    printf("sumk1 : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[sumk1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumk1_st]);
    printf("lap : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[lap_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[lap_st]);

    std::fill(timings.begin(),timings.end(), 0);
    launch_ldg<double>(timings, isize, jsize, ksize, tsteps, warmup_step);

    printf("-------------   TEXTURE   -------------\n");
    printf("copy : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[copy_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[copy_st]);
    printf("copyi1 : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[copyi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[copyi1_st]);
    printf("sumi1 : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[sumi1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumi1_st]);
    printf("sumj1 : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[sumj1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumj1_st]);
    printf("sumk1 : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[sumk1_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[sumk1_st]);
    printf("lap : %f GB/s, time : %f \n", tot_size*2*sizeof(double)/(timings[lap_st]/(double)(tsteps - (warmup_step+1)))/(1024.*1024.*1024.), timings[lap_st]);

}
