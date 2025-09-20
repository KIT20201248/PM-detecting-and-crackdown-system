cmake_minimum_required(VERSION 3.5)
project(nav2_custom)

# 필수 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav2_bringup REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)

include_directories(include)


# 소스 추가
add_library(sidewalk_layer SHARED
  src/costmap_plugins/sidewalk_layer.cpp
)

ament_target_dependencies(sidewalk_layer
  rclcpp
  nav2_costmap_2d
  pluginlib
)

# 플러그인 xml 등록
pluginlib_export_plugin_description_file(nav2_costmap_2d plugins.xml)

# Launch 및 Params 디렉토리 설치
install(
  DIRECTORY launch params
  DESTINATION share/${PROJECT_NAME}
)

# 플러그인 설치
install(
  TARGETS sidewalk_layer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 헤더 설치
install(
  DIRECTORY include/
  DESTINATION include
)

# plugins.xml 설치
install(
  FILES plugins.xml
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  src/nav2_custom/waypoint_sender.py
  DESTINATION lib/${PROJECT_NAME}
)


ament_package()
