cmake_minimum_required(VERSION 3.10)
project(tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# 命令行参数
option(ENABLE_TESTS "Build tests" OFF)

if(EXISTS "/opt/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc")
    set(CROSS_COMPILE "/opt/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-")
else()
    set(CROSS_COMPILE "aarch64-linux-gnu-")
endif()

# Set cross-compilation toolchain
set(CMAKE_C_COMPILER "${CROSS_COMPILE}gcc")
set(CMAKE_CXX_COMPILER "${CROSS_COMPILE}g++")
set(CMAKE_AR "${CROSS_COMPILE}ar")
set(CMAKE_AS "${CROSS_COMPILE}as")
set(CMAKE_LD "${CROSS_COMPILE}ld")
set(CMAKE_STRIP "${CROSS_COMPILE}strip")
set(CMAKE_RANLIB "${CROSS_COMPILE}ranlib")
set(CMAKE_OBJCOPY "${CROSS_COMPILE}objcopy")
set(CMAKE_OBJDUMP "${CROSS_COMPILE}objdump")
set(CMAKE_READELF "${CROSS_COMPILE}readelf")
set(CMAKE_NM "${CROSS_COMPILE}nm")

# Set compiler flags
add_compile_options(-Werror) # remove Werror
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Equivalent to -fPIC
set(CMAKE_CXX_STANDARD 17)

# Define paths
set(HR_TOP_DIR "/usr/hobot")
if(NOT DEFINED HR_BUILD_OUTPUT_DIR)
    set(HR_BUILD_OUTPUT_DIR ${HR_TOP_DIR})
endif()

if(NOT DEFINED APP_DEPLOY_DIR)
    set(APP_DEPLOY_DIR "/app/")
endif()

set(PLATFORM_SAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PLATFORM_SAMPLES_DEPLOY_DIR "${APP_DEPLOY_DIR}/platform_samples")

# Enable verbose output based on BUILD_VERBOSE
option(BUILD_VERBOSE "Enable verbose build output" OFF)
if(BUILD_VERBOSE)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif()

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 添加包含目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "${APP_DEPLOY_DIR}/multimedia_samples/mipicam/mipicam"
    "${APP_DEPLOY_DIR}/multimedia_samples/utils"
    "${APP_DEPLOY_DIR}/multimedia_samples/vp_sensors"
    "include"
)

link_directories(
    "${APP_DEPLOY_DIR}/multimedia_samples/mipicam/mipicam"
    ${PLATFORM_SAMPLES_DIR}/lib
    /usr/hobot/lib
)

# 添加可执行文件
add_executable(CameraServer 
    src/CameraServer.cpp
    src/tracker.cpp)
if (ENABLE_TESTS)
    add_executable(tmp_server src/tmp_server.cpp)
    add_executable(tmp_receive src/tmp_receive.cpp)
    add_executable(tmp_multi_server src/tmp_multi_server.cpp)
    add_executable(tmp_visualize_send src/tmp_visualize_send.cpp)
    add_executable(tmp_http src/tmp_http.cpp)
endif()

target_link_libraries(CameraServer
    ${OpenCV_LIBS}
    mipicam
    cam
    vpf
    hbmem
    gdcbin
    cjson
    pthread
    alog
    dl
    curl
)

if (ENABLE_TESTS)
    target_link_libraries(tmp_server
        ${OpenCV_LIBS}
        )
    target_link_libraries(tmp_receive
        ${OpenCV_LIBS}
        )
    target_link_libraries(tmp_multi_server
        ${OpenCV_LIBS}
        )
    target_link_libraries(tmp_visualize_send
        ${OpenCV_LIBS}
        )
    target_link_libraries(tmp_http
        ${OpenCV_LIBS}
        curl
        )
endif()

