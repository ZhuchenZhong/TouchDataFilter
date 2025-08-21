cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(TouchDataFilter)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置LibTorch路径
list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")

# 查找必要的依赖项
find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "Torch库未找到，请确认安装路径是否正确")
endif()

# 寻找Python组件
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(NumPy REQUIRED)

# 设置包含路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 收集源文件
set(PREDICTOR_SOURCES "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/TouchModelPredictor.cpp")
    list(APPEND PREDICTOR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/include/TouchModelPredictor.cpp")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/TouchModelPredictor.cpp")
    list(APPEND PREDICTOR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/TouchModelPredictor.cpp")
else()
    message(WARNING "未找到TouchModelPredictor实现文件，请确保它存在或创建")
endif()

# 定义库的源文件
set(LIB_SOURCES ${PREDICTOR_SOURCES})

# 创建静态库
add_library(TouchDataFilter_static STATIC ${LIB_SOURCES})
set_target_properties(TouchDataFilter_static PROPERTIES 
    OUTPUT_NAME "TouchDataFilter"
    CXX_STANDARD 17
)
target_link_libraries(TouchDataFilter_static "${TORCH_LIBRARIES}")

# 创建动态库
add_library(TouchDataFilter_shared SHARED ${LIB_SOURCES})
set_target_properties(TouchDataFilter_shared PROPERTIES 
    OUTPUT_NAME "TouchDataFilter"
    CXX_STANDARD 17
    VERSION 1.0.0
    SOVERSION 1
)
target_link_libraries(TouchDataFilter_shared "${TORCH_LIBRARIES}")

# 创建预测程序
add_executable(predict predict.cpp)
target_link_libraries(predict TouchDataFilter_static "${TORCH_LIBRARIES}")
set_property(TARGET predict PROPERTY CXX_STANDARD 17)

# 创建测试程序
add_executable(test_model_cpp ./test/test_model_cpp.cpp)
target_link_libraries(test_model_cpp TouchDataFilter_static "${TORCH_LIBRARIES}")
set_property(TARGET test_model_cpp PROPERTY CXX_STANDARD 17)

# 使用动态库编译
add_custom_target(
    cython_wrapper
    COMMAND ${Python3_EXECUTABLE} -m cython -3 --embed -o ${CMAKE_CURRENT_BINARY_DIR}/predict_wrapper.c ${CMAKE_CURRENT_SOURCE_DIR}/predict_wrapper.pyx
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating C code from Cython"
)

# 添加库目标
add_library(predict_wrapper SHARED ${CMAKE_CURRENT_BINARY_DIR}/predict_wrapper.c)
add_dependencies(predict_wrapper cython_wrapper)

# 设置包含目录和链接库
target_include_directories(predict_wrapper PRIVATE 
    ${Python3_INCLUDE_DIRS}
    ${NumPy_INCLUDE_DIRS}
)

target_link_libraries(predict_wrapper
    ${Python3_LIBRARIES}
    ${TORCH_LIBRARIES}
)

# 复制模型文件到构建目录
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/models)
file(GLOB MODEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/models/*/*.pth")
foreach(MODEL_FILE ${MODEL_FILES})
    get_filename_component(FILE_NAME ${MODEL_FILE} NAME)
    configure_file(${MODEL_FILE} ${CMAKE_BINARY_DIR}/models/${FILE_NAME} COPYONLY)
endforeach()

# 安装规则
install(TARGETS predict TouchDataFilter_static TouchDataFilter_shared
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(TARGETS predict_wrapper
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# 安装头文件
install(FILES include/TouchData.hpp include/TouchModelPredictor.hpp
        DESTINATION include/TouchDataFilter)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/predict_wrapper.h
        DESTINATION include/TouchDataFilter)

message(STATUS "LibTorch库路径: ${TORCH_LIBRARIES}")
message(STATUS "LibTorch包含路径: ${TORCH_INCLUDE_DIRS}")
message(STATUS "C++标准: ${CMAKE_CXX_STANDARD}")