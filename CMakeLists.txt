cmake_minimum_required(VERSION 3.20)
project(MiniFaiss CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -pthread")

include_directories(include)

add_library(vector_engine 
	src/main.cpp
	src/index.cpp
)

add_executable(mini_faiss src/main.cpp)

target_link_libraries(mini_faiss PRIVATE vector_engine)
