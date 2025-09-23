cmake_minimum_required(VERSION 3.11)

project(libjson)

set(HEADERS
        nlohmann/json.hpp # https://github.com/nlohmann/json/blob/efcf9efb4f804fedb0a66f5218c850198d616637/single_include/nlohmann/json.hpp
        )

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR})
