target_sources(
  ${PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csv_reader.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dsg.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/frontier_utils.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/nearest_neighbor.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/offscreen_renderer.cpp
)
