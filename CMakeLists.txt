cmake_minimum_required(VERSION 3.20)

project(facefusion-onnxrun CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_HIGH_ONNX_RUNTIME "Use the high version onnxruntime" ON)
option(USE_CUDA_BACKEND "Using cuda backend" OFF)
option(USE_OPENCV_HIGHGUI "Using opencv highgui" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install CACHE PATH "Force path to set CMAKE_INSTALL_PREFIX" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v3.0
)
FetchContent_MakeAvailable(argparse)

if(USE_OPENCV_HIGHGUI)
    find_package(OpenCV REQUIRED
        COMPONENTS
        core
        imgcodecs
        imgproc
        calib3d
        highgui)
else()
    find_package(OpenCV REQUIRED
        COMPONENTS
        core
        imgcodecs
        imgproc
        calib3d)
endif()



message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(onnxruntime REQUIRED)
message(STATUS "onnxruntime library status:")
message(STATUS "    version: ${onnxruntime_VERSION}")

file(GLOB SRC_SRCS ./cpp/*.h ./cpp/*.hpp ./cpp/*.cpp)

add_executable(${PROJECT_NAME} ${SRC_SRCS})

if(USE_HIGH_ONNX_RUNTIME)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DUSE_HIGH_ONNX_RUNTIME)
endif()

if(USE_CUDA_BACKEND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DUSE_CUDA_BACKEND)
endif()

if(USE_OPENCV_HIGHGUI)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DUSE_OPENCV_HIGHGUI)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE 
    ./cpp 
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${OpenCV_LIBS}
    argparse
    onnxruntime::onnxruntime
)

if(APPLE)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    MACOSX_BUNDLE FALSE
    XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY ""
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Manual"
  )
elseif(UNIX)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})
endif()

# add_custom_command(TARGET ${PROJECT_NAME}
#     POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/weights $<TARGET_FILE_DIR:${PROJECT_NAME}>/weights
#     COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/cpp/model_matrix.bin $<TARGET_FILE_DIR:${PROJECT_NAME}>/weights/model_matrix.bin
#     COMMENT "Copying weight files")

file(COPY ${CMAKE_CURRENT_LIST_DIR}/weights 
    DESTINATION ${CMAKE_INSTALL_PREFIX})
file(COPY ${CMAKE_CURRENT_LIST_DIR}/cpp/model_matrix.bin 
    DESTINATION ${CMAKE_INSTALL_PREFIX}/weights)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    BUNDLE DESTINATION bin
)

# install(DIRECTORY
#     ${CMAKE_CURRENT_LIST_DIR}/weights
#     DESTINATION ${CMAKE_INSTALL_PREFIX}
# )

# install(FILES
#     ${CMAKE_CURRENT_LIST_DIR}/cpp/model_matrix.bin
#     DESTINATION weights
# )