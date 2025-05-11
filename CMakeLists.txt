cmake_minimum_required(VERSION 3.21)
project(Gomoku)


set(CMAKE_CXX_STANDARD 20)
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
list(APPEND CMAKE_AUTOUIC_SEARCH_PATHS "${PROJECT_SOURCE_DIR}/ui")

find_package(Qt6 COMPONENTS Core Widgets REQUIRED)
file(GLOB SRC "source/*.cpp")
file(GLOB INC "include/*.h")

add_executable(Gomoku ${SRC} ${INC})
target_include_directories(Gomoku PRIVATE include)
target_link_libraries(Gomoku PRIVATE
        Qt6::Core
        Qt6::Widgets)

