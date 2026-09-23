cmake_minimum_required(VERSION 3.14)
project(hvlm_planner VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_compile_options(-Wall -Wextra)

find_package(config_utilities REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spark_dsg REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nanoflann REQUIRED)
find_package(Open3D REQUIRED)

add_library(${PROJECT_NAME})
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC config_utilities::config_utilities
         Eigen3::Eigen
         spark_dsg::spark_dsg
         nanoflann::nanoflann
         Open3D::Open3D
         ${OpenCV_LIBRARIES}
)

set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(hvlm_planner::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

add_subdirectory(src)

include(GNUInstallDirs)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT hvlm_planner-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT hvlm_planner-targets FILE hvlm_plannerTargets.cmake NAMESPACE hvlm_planner::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hvlm_planner
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/hvlm_plannerConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/hvlm_plannerConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hvlm_planner
)
write_basic_package_version_file(
  hvlm_plannerConfigVersion.cmake VERSION ${PACKAGE_VERSION} COMPATIBILITY AnyNewerVersion
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/hvlm_plannerConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/hvlm_plannerConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hvlm_planner
)

find_package(ament_cmake_core QUIET)
if (${ament_cmake_core_FOUND})
  # NOTE(nathan) this will actually likely change once we drop ros1 support, but
  # for now this is a good proxy if we are in ros2 and need to install the config files
  # versus in ros1 (where installing them breaks things)
  install(DIRECTORY config DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME})
  ament_package()
endif()
