
# needs editing 
FROM ghcr.io/smart-wheelchair-rrc/wheelchair2_base:v1.0
# Add the following labels
LABEL org.opencontainers.image.description="Crowdsurfer simulation" 
LABEL org.opencontainers.image.title="Wheelchair2Navigation"
LABEL org.opencontainers.image.vendor="Soumo Roy"
LABEL org.opencontainers.image.source="https://github.com/s0um0r0y/crowdsurfer-ros1"
LABEL maintainer="soumoroy09@gmail.com"
LABEL org.opencontainers.image.licenses="MIT"

COPY . /home/crowdsurfer_ros1/src

WORKDIR /home/crowdsurfer_ros1/
# handle default shell
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/noetic/setup.bash \
    # might throw error need to see ROS1 version
    && catkin_make --cmake-args -DROS_EDITION=ROS1 -DHUMBLE_ROS=noetic \
    && catkin_make --packages-select Crowdsurfer pedsim_ros_with_gazebo turtlebot2_noetic_packages \
    && source install/setup.bash \
    && cd src/crowdsurfer_ros1/launch \
    && sudo chmod +x ./run_closed_loop_simulation.sh
# need to mount the local directory to /workspace