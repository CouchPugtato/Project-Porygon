cmake_minimum_required(VERSION 3.10)
project(ShowdownAI)

set(CMAKE_C_STANDARD 99)

add_executable(ShowdownAI
    src/main.c
    src/pokemon.c
)