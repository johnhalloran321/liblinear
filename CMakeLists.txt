cmake_minimum_required (VERSION 2.6)
project (liblinear)
find_package(CUDA 9.0 REQUIRED)
find_package(OpenMP REQUIRED)

find_cuda_helper_libs(cusparse)
set(CUDA_CUSPARSE_LIBRARIES ${CUDA_cusparse_LIBRARY})
macro(CUDA_ADD_CUSPARSE_TO_TARGET target)
     target_link_libraries(${target} ${CUDA_cusparse_LIBRARY})
endmacro()

    
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3 ")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3")
set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-D_FORCE_INLINES -lineinfo) 
set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)
set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35)
set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52)
		
set (CUDA_SAMPLES_DIR "${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc")

include_directories(blas)
add_subdirectory(blas)

include_directories("${CUDA_SAMPLES_DIR}")
include_directories(".")


# executable	
cuda_add_executable(train ./train.c ./linear.cpp ./tron.cu)
CUDA_ADD_CUBLAS_TO_TARGET (train)
CUDA_ADD_CUSPARSE_TO_TARGET (train)
    
target_link_libraries(train blas)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

install (TARGETS train DESTINATION .)

		
		
