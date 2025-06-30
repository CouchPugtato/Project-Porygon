cmake_minimum_required(VERSION 3.10)
project(ShowdownAI)

set(CMAKE_C_STANDARD 99)


add_executable(ShowdownAI
    src/main.c
    src/pokemon.c
    src/mongoose.c
)

target_link_libraries(ShowdownAI PRIVATE ws2_32)
target_compile_definitions(ShowdownAI PRIVATE _CRT_RAND_S)