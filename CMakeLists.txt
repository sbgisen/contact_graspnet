cmake_minimum_required(VERSION 3.0.2)
project(contact_graspnet_ros)

find_package(
  catkin REQUIRED
  COMPONENTS sensor_msgs
             geometry_msgs
             std_msgs
             cv_bridge
             rospy
             message_generation
)

add_service_files(FILES Inference.srv)

generate_messages(DEPENDENCIES sensor_msgs geometry_msgs std_msgs)

catkin_package(
  CATKIN_DEPENDS
    sensor_msgs
    geometry_msgs
    std_msgs
    cv_bridge
    rospy
    message_runtime
)
