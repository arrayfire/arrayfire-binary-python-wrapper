cmake_minimum_required(VERSION 3.20)

project(
  ${SKBUILD_PROJECT_NAME}
  LANGUAGES C CXX
  VERSION ${SKBUILD_PROJECT_VERSION})

find_package(Python COMPONENTS Interpreter Development.Module)


if(DEFINED ENV{AF_BUILD_LOCAL_LIBS})
    include(FetchContent)
    
    #set(NO_SONAME)
    FetchContent_Declare(
        arrayfire
        GIT_REPOSITORY https://github.com/edwinsolisf/arrayfire.git
        GIT_TAG afwheel310
    )
    #TODO: change package name to match repository/project name?
    #set(AF_INSTALL_CMAKE_DIR "${SKBUILD_PROJECT_NAME}")
    
    set(AF_INSTALL_BIN_DIR "arrayfire_wrapper/binaries")
    set(AF_INSTALL_CMAKE_DIR "arrayfire_wrapper/binaries")
    set(AF_INSTALL_DOC_DIR "arrayfire_wrapper/binaries")
    set(AF_INSTALL_EXAMPLE_DIR "arrayfire_wrapper/binaries")
    set(AF_INSTALL_INC_DIR "arrayfire_wrapper/binaries")
    set(AF_INSTALL_LIB_DIR "arrayfire_wrapper/binaries")
    set(FG_INSTALL_LIB_DIR "arrayfire_wrapper/binaries")

    FetchContent_MakeAvailable(arrayfire)
endif()

# Testing artifacts
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/generated.txt "Testing")
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/generated.txt
  DESTINATION arrayfire_wrapper
  COMPONENT Generated)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/generated_ignored.txt "Testing")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated_ignored.txt
        DESTINATION arrayfire_wrapper)
