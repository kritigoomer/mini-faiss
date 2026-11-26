cmake_minimum_required(VERSION 3.20)
project(mini_faiss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(mini_faiss_lib STATIC
    src/knn_index.cpp
)

target_include_directories(mini_faiss_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Enable high optimization and native SIMD features where supported
if(MSVC)
    target_compile_options(mini_faiss_lib PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(mini_faiss_lib PRIVATE -O3)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx2 -mfma" HAS_AVX2_FMA)
        if(HAS_AVX2_FMA)
            target_compile_options(mini_faiss_lib PRIVATE -mavx2 -mfma)
        endif()
    endif()
endif()

add_executable(mini_faiss_bench src/main.cpp)
target_link_libraries(mini_faiss_bench PRIVATE mini_faiss_lib)
