set(HEADERS mesh.hpp stencil_kernels.h ../tools.h array.hpp)

cuda_add_executable (stencil_unstructured_bench main.cu ${HEADERS} OPTIONS "-std=c++11")
target_link_libraries(stencil_unstructured_bench json)

