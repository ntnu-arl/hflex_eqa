target_sources(
  ${PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/agent_state.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/exploration_bounding_box.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/global_info.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hflex_eqa_pipeline.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/input.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/label_remapper.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/label_space_config.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_queues.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/robot_prefix_config.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/semantic_color_map.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
)
