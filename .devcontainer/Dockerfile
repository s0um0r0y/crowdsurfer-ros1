
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
    && catkin_make \
    && source devel/setup.bash
    # && sudo chmod +x ./run_closed_loop_simulation.sh
# need to mount the local directory to /workspace