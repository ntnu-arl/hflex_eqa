target_sources(
  ${PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/a_star.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cubic_spline_planner.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bspline_planner.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dijkstra.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/linear_interpolator_planner.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/low_level_module.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/node_search.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/planner.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/semantic_search.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shortest_path.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/clip_search.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/find_room_search.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/question_room_clip_search.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/closest_search.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/object_vps_search.cpp
)
