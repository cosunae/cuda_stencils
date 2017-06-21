set(HEADERS mesh.hpp stencil_kernels.h ../tools.h array.hpp udefs.hpp helpers.hpp)

cuda_add_executable (stencil_unstructured_bench main.cu ${HEADERS} )
target_link_libraries(stencil_unstructured_bench json)

add_executable (mesh_test test.cpp ${HEADERS})
target_link_libraries(mesh_test json)

