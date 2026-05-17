cmake_minimum_required(VERSION 3.16)
project(showdownAITest C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(LibWebSockets REQUIRED)
find_package(CURL REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(showdown_client
	src/main.c
	src/showdown_client.c
	src/neural_net.c
	src/game_state.c
	src/observation.c
	src/episode.c
	src/gru_model.c
)

target_link_libraries(showdown_client PRIVATE websockets)
target_link_libraries(showdown_client PRIVATE CURL::libcurl)

# Link math library when building on Unix-like systems (required for log/cos/sqrt)
if(UNIX)
    target_link_libraries(showdown_client PRIVATE m)
endif()
