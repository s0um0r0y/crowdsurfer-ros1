# Install script for directory: /home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/src/turtlebot2_noetic_packages/turtlebot_simulator/turtlebot_stdr

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/build/turtlebot2_noetic_packages/turtlebot_simulator/turtlebot_stdr/catkin_generated/installspace/turtlebot_stdr.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/turtlebot_stdr/cmake" TYPE FILE FILES
    "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/build/turtlebot2_noetic_packages/turtlebot_simulator/turtlebot_stdr/catkin_generated/installspace/turtlebot_stdrConfig.cmake"
    "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/build/turtlebot2_noetic_packages/turtlebot_simulator/turtlebot_stdr/catkin_generated/installspace/turtlebot_stdrConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/turtlebot_stdr" TYPE FILE FILES "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/src/turtlebot2_noetic_packages/turtlebot_simulator/turtlebot_stdr/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/etc/catkin/profile.d" TYPE FILE FILES "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/build/turtlebot2_noetic_packages/turtlebot_simulator/turtlebot_stdr/catkin_generated/installspace/25.turtlebot-stdr.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/turtlebot_stdr/catkin_env_hook" TYPE FILE FILES "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/build/turtlebot2_noetic_packages/turtlebot_simulator/turtlebot_stdr/catkin_generated/installspace/25.turtlebot-stdr.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/turtlebot_stdr" TYPE DIRECTORY FILES "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/src/turtlebot2_noetic_packages/turtlebot_simulator/turtlebot_stdr/launch")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/turtlebot_stdr" TYPE DIRECTORY FILES "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/src/turtlebot2_noetic_packages/turtlebot_simulator/turtlebot_stdr/maps")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/turtlebot_stdr" TYPE DIRECTORY FILES "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/src/turtlebot2_noetic_packages/turtlebot_simulator/turtlebot_stdr/rviz")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/turtlebot_stdr" TYPE DIRECTORY FILES "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/src/turtlebot2_noetic_packages/turtlebot_simulator/turtlebot_stdr/robot")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/turtlebot_stdr" TYPE PROGRAM FILES "/home/soumoroy/crowdsurfer_new_implementation_ws/crowdsurfer-ros1/src/turtlebot2_noetic_packages/turtlebot_simulator/turtlebot_stdr/nodes/tf_connector.py")
endif()

