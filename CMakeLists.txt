cmake_minimum_required(VERSION 3.16)
project(showdownAITest C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(BUILD_LEGACY_NATIVE_SHOWDOWN "Build the old native C Showdown websocket client" OFF)

if(BUILD_LEGACY_NATIVE_SHOWDOWN)
    find_package(LibWebSockets REQUIRED)
    find_package(CURL REQUIRED)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SHOWDOWN_CLIENT_SOURCES
	src/main.c
	src/neural_net.c
	src/game_state.c
	src/observation.c
	src/episode.c
	src/gru_model.c
	src/id_tables.c
	src/state_value.c
	src/request_parser.c
	src/action_mapper.c
	src/raw_battle_state.c
	src/event_parser.c
	src/inference_engine.c
	src/observation_builder.c
	src/runtime_protocol.c
	src/replay_io.c
	src/env_session.c
	src/checkpoint.c
	src/gru_trainer.c
)

if(BUILD_LEGACY_NATIVE_SHOWDOWN)
    list(APPEND SHOWDOWN_CLIENT_SOURCES src/showdown_client.c)
endif()

add_executable(showdown_client ${SHOWDOWN_CLIENT_SOURCES})

set(SHOWDOWN_LEGALITY_TEST_SOURCES
	src/legality_tests.c
	src/request_parser.c
	src/action_mapper.c
	src/id_tables.c
)

add_executable(showdown_legality_tests ${SHOWDOWN_LEGALITY_TEST_SOURCES})

if(BUILD_LEGACY_NATIVE_SHOWDOWN)
    target_compile_definitions(showdown_client PRIVATE HAVE_NATIVE_SHOWDOWN_CLIENT=1)
    target_link_libraries(showdown_client PRIVATE websockets)
    target_link_libraries(showdown_client PRIVATE CURL::libcurl)
endif()

# Link math library when building on Unix-like systems (required for log/cos/sqrt)
if(UNIX)
    target_link_libraries(showdown_client PRIVATE m)
    target_link_libraries(showdown_legality_tests PRIVATE m)
endif()
