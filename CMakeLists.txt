cmake_minimum_required(VERSION 3.21)
project(Gomoku CXX CUDA)

# 基础设置
set(CMAKE_CXX_STANDARD 20)
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

# QtMOC设置
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
list(APPEND CMAKE_AUTOUIC_SEARCH_PATHS "${PROJECT_SOURCE_DIR}/ui")

# 环境依赖设置
find_package(Qt6 COMPONENTS Core Widgets REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "86")
find_package(CUDAToolkit REQUIRED)

# 文件检索
file(GLOB SRC "${PROJECT_SOURCE_DIR}/source/*.cpp")
file(GLOB INC "${PROJECT_SOURCE_DIR}/include/*.h")
message("${INC}")

# 编译动态链接库
file(GLOB CuSRC "${PROJECT_SOURCE_DIR}/cuda/*")
add_library(CuRoll SHARED ${CuSRC})
target_include_directories(CuRoll PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_options(CuRoll PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
set_target_properties(CuRoll PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 编译源文件链接动态库
add_executable(Gomoku ${SRC} ${INC})
target_include_directories(Gomoku PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(
        Gomoku PRIVATE
        Qt6::Core
        Qt6::Widgets
        CuRoll
)
