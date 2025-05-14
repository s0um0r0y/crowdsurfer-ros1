
# needs editing 
# FROM ghcr.io/smart-wheelchair-rrc/crowdsurfer_base:v1.0 
# # Add the following labels
# # needs to be removed ig 
# LABEL org.opencontainers.image.description="Crowdsurfer simulation" 
# LABEL org.opencontainers.image.title="Wheelchair2Navigation"
# LABEL org.opencontainers.image.vendor="Soumo Roy"
# LABEL org.opencontainers.image.source="https://github.com/s0um0r0y/crowdsurfer-ros1"
# LABEL maintainer="soumoroy09@gmail.com"
# LABEL org.opencontainers.image.licenses="MIT"

# from crowdsurfer_ros1 something like that needs to be done
FROM crowdsurfer

COPY . /home/crowdsurfer_ros1/src

WORKDIR /home/crowdsurfer_ros1/
# handle default shell
SHELL ["/bin/bash", "-c"]

# Install dependencies if not already in base image
RUN sudo apt-get update && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y

RUN source /opt/ros/noetic/setup.bash \
    # might throw error need to see ROS1 version
    && catkin_make \
    && catkin_make --packages-select Crowdsurfer pedsim_ros_with_gazebo turtlebot2_noetic_packages \
    && source install/setup.bash \
    && cd src/crowdsurfer_ros1/ \
    && source devel/setup.bash
    # && sudo chmod +x ./run_closed_loop_simulation.sh
# need to mount the local directory to /workspace