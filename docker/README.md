## Hierarchy of Dockerfiles to be Built

1. **noetic_cuda11.8_x86** (base image: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04`)  
   *Call file name: `cuda_ros_docker`*  
   - Sets up the CUDA environment with ROS Noetic.
   - Installs Python, pip, ROS Noetic packages, and other basic utilities.
   - Prepares the container with the ros_entrypoint.sh for launching ROS.
   
   Docker build command for x86:  
   `docker build -t cuda_ros_docker -f crowdsurfer_ros1/docker/noetic_cuda11.8_x86/dockerfile crowdsurfer_ros1/docker/noetic_cuda11.8_x86`

3. **ros1_crowdsurfer_base_x86** (base image: `cuda_ros_docker`)  
   *Call file name: `crowdsurfer_base`*  
   - Extends the CUDA-based image to include additional dependencies.
   - Installs extra packages and Python libraries needed for the Crowdsurfer base.
   - Serves as the foundation for further ROS1 development for crowdsurfer applications.
   
   Docker build command for x86:  
   `docker build -t crowdsurfer_base -f crowdsurfer_ros1/docker/ros1_crowdsurfer_base_x86/dockerfile crowdsurfer_ros1/docker/ros1_crowdsurfer_base_x86`

4. **crowdsurfer_ros1_x86** (base image: `crowdsurfer_base`)  
   *Call file name: `crowdsurfer`*  
   - Clones and sets up robotics and simulation repositories.
   - Installs tools like gdown and additional ROS packages.
   - Configures a workspace for building and running the Crowdsurfer ROS1 application.
   
   Docker build command for x86:  
   `docker build -t crowdsurfer -f crowdsurfer_ros1/docker/crowdsurfer_ros1_x86/dockerfile crowdsurfer_ros1/docker/crowdsurfer_ros1_x86`

5. **wheelchair1_crowdsurfer_x86** (base image: `crowdsurfer`)  
   *Call file name: `wheelchair1_crowdsurfer`*  
   - Builds on the Crowdsurfer container by integrating hardware drivers.
   - Installs and compiles Librealsense and Livox SDK2.
   - Clones and configures the wheelchair camera–lidar application.
   - Prepares the final image for running the complete wheelchair integration.
   
   Docker build command for x86:  
   `docker build -t wheelchair1_crowdsurfer -f crowdsurfer_ros1/docker/wheelchair1_crowdsurfer_x86/dockerfile crowdsurfer_ros1/docker/wheelchair1_crowdsurfer_x86`

### Running the container
```
docker run -it –rm --privileged --cap-add=SYS_NICE --ulimit rtprio=99 --ulimit rttime=-1 --ulimit memlock=8428281856 --cap-add=all --security-opt seccomp:unconfined –security-opt apparmor:unconfined --volume=/dev:/dev --net=host --ipc=host -e DISPLAY=$DISPLAY -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY}"  -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR}" -e PULSE_SERVER="${PULSE_SERVER}" -e QT_X11_NO_MITSHM="1"  -e LIBGL_ALWAYS_SOFTWARE="1"  --device /dev/ttyUSB0:/dev/ttyUSB0 --entrypoint /bin/bash --name wheelchair_cs ``
