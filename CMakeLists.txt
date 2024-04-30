cmake_minimum_required(VERSION 3.8)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(SosoNoWPT LANGUAGES CXX CUDA)
project(SosoNoWPT CXX C CUDA)
#set(CMAKE_CUDA_ARCHITECTURES 52;70;75;86)
set(CMAKE_CXX_STANDARD 20)
#-------------------------------------------------------------------------------
# Set target
#-------------------------------------------------------------------------------
set(APPLICATION_3_1_SOURCE

        # Scene Object & Structure
        src/scene/bvh.cu

        # Pathtracer
        src/pathtracer/camera.cu
        src/pathtracer/pathtracer.cu

        src/scene/shape.h
        src/scene/media.cuh
        src/scene/microfacet.cuh
)

set(APPLICATION_3_2_SOURCE

        # MeshEdit

        src/pathtracer/raytraced_renderer.cu

        # misc
        src/util/lodepng.cpp

        # Application
        src/util/device-utils.h
        src/util/properties.h
        src/util/device-vec-ops.h
        src/util/image.cu
        src/apps/wpt.cu
        src/util/obj-io.cc
        src/util/obj-io.h
        src/util/medium-io.h
)

set(APPLICATION_HEADERS
        # Scene Object & Structure
        src/scene/aggregate.h
        src/scene/bbox.h
        src/scene/bvh.h
        src/scene/object.h
        src/scene/primitive.h
        src/scene/triangle.h
        src/scene/sphere.h
        src/scene/scene.h
        # MeshEdit
        src/util/image.h
        src/util/random_util.h
        src/util/device-utils.h
        src/util/device-vec-ops.h
        # Pathtracer
        src/pathtracer/camera.h
        src/pathtracer/pathtracer.h
        src/pathtracer/raytraced_renderer.h
        src/pathtracer/sampler.h
        # misc
        src/util/lodepng.h
        # Application
        src/util/gpu-textures.h
)

if (WIN32)
    list(APPEND APPLICATION_3_1_SOURCE src/util/win32/getopt.c)
endif ()

if (WIN32 OR APPLE)
    add_library(pt31 STATIC ${APPLICATION_3_1_SOURCE} ${APPLICATION_HEADERS})
else ()
    add_library(pt31 SHARED ${APPLICATION_3_1_SOURCE} ${APPLICATION_HEADERS})
endif ()

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

target_include_directories(pt31 PUBLIC src)

add_executable(pathtracer ${APPLICATION_3_2_SOURCE} ${APPLICATION_HEADERS})
target_include_directories(pathtracer PUBLIC src)

target_link_libraries(pt31 PUBLIC CGL Thrust)
target_link_libraries(pathtracer PUBLIC pt31)

set(CGL_INCLUDE_DIRS CGL/include CGL/deps/glew/include CGL/deps/glfw/include ./src/imgui ./src/imgui/backends)

#-------------------------------------------------------------------------------
# Find dependencies
#-------------------------------------------------------------------------------
add_subdirectory(CGL)
target_include_directories(pt31 PUBLIC ${CGL_INCLUDE_DIRS})
target_include_directories(pathtracer PUBLIC ${CGL_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CGL_CXX_FLAGS}")

set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
target_link_libraries(pt31 PUBLIC OpenGL::GL)
target_link_libraries(pt31 PUBLIC OpenGL::GLU)
