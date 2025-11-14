cmake_minimum_required(VERSION 3.16)
project(showdownAITest C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(LibWebSockets REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(showdown_client
	src/main.c
	src/showdown_client.c
	src/battle_ai.c
)

target_link_libraries(showdown_client PRIVATE websockets)

