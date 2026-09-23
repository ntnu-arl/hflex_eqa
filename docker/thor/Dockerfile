# Base image with CUDA and PyTorch, ubuntu 24.04
FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy

#setup ros2 workspace (mounted from host)
WORKDIR /developer

#ros2 jazzy
RUN apt-get update && \
    apt-get install -y curl gnupg2 lsb-release software-properties-common && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /etc/apt/trusted.gpg.d/ros.gpg && \
    echo "deb http://packages.ros.org/ros2/ubuntu noble main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && \
    apt-get install -y \
    ros-${ROS_DISTRO}-desktop \
    python3-rosdep \
    python3-colcon-common-extensions && \
    apt-get install -y \
    ros-${ROS_DISTRO}-depth-image-proc \
    ros-${ROS_DISTRO}-rqt-tf-tree \
    ros-${ROS_DISTRO}-rmw-zenoh-cpp  \
    ros-${ROS_DISTRO}-pcl-ros     \
    ros-${ROS_DISTRO}-tf2-eigen   \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp && \
    rm -rf /var/lib/apt/lists/*

RUN rosdep init && rosdep update

#install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    usbutils \
    git \
    curl \
    wget \
    tmux \
    vim \
    htop \
    net-tools \
    usbutils \
    nlohmann-json3-dev \
    libgoogle-glog-dev \
    libcli11-dev \
    libeigen3-dev \
    libzmq5-dev \
    ros-dev-tools \
    nlohmann-json3-dev \
    libgoogle-glog-dev \
    libcli11-dev \
    libeigen3-dev \
    libjsoncpp-dev \
    libspdlog-dev \
    libcurl4-openssl-dev \
    libnanoflann-dev && \
    rm -rf /var/lib/apt/lists/*

# Realsense sdk
RUN mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.realsenseai.com/Debian/librealsenseai.asc | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/librealsenseai.gpg > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo `lsb_release -cs` main" | \
    tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && apt-get install -y librealsense2-utils librealsense2-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    libunwind-dev\
    libjpeg-dev \
    libglm-dev \
    mesa-utils \
    libglu1-mesa-dev \
    freeglut3-dev \
    xorg-dev \
    g++-11 \
    gcc-11 && \
    rm -rf /var/lib/apt/lists/*

#install pipx + rosbags
RUN apt-get update && apt-get install -y pipx && \
    pipx install rosbags && \
    echo 'export PATH=$PATH:/root/.local/bin' >> /root/.bashrc && \
    rm -rf /var/lib/apt/lists/*

#networking
RUN apt update && apt install -y ros-jazzy-rmw-zenoh-cpp \
    python3-colcon-clean libdxflib-dev

# Install Open3D
# RUN git clone --branch v0.19.0 https://github.com/isl-org/Open3D.git --depth 1 && \
#     cd Open3D && \
#     util/install_deps_ubuntu.sh assume-yes && \
#     mkdir build && cd build && \
#     cmake .. && \
#     find . -path '*filament*ImageSampler.cpp' -exec sed -i 's/\bM_PIf\b/M_PI_f/g' {} \; && \
#     make -j$(nproc) && make install

RUN git clone --branch v0.19.0 https://github.com/isl-org/Open3D.git --depth 1 && \
    cd Open3D && \
    util/install_deps_ubuntu.sh assume-yes && \
    mkdir -p build && cd build && \
    cmake .. && \
    (make ext_filament -j$(nproc) || make -j$(nproc) || true) && \
    FILE="$(find /developer/Open3D -path '*filament*' -name ImageSampler.cpp | head -n 1)" && \
    test -n "$FILE" && \
    sed -i 's/constexpr float M_PIf/constexpr float M_PI_F_FILAMENT/g; s/\bM_PIf\b/M_PI_F_FILAMENT/g' "$FILE" && \
    grep -n "M_PI" "$FILE" && \
    make -j$(nproc) && \
    make install && \
    mkdir -p /developer/share/Open3D && \
    ln -s /usr/local/bin/Open3D/resources /developer/share/Open3D/resources

#copy source code for rosdep (replaced by a volume mount in container)
COPY src/ /tmp/deps/src/

#install dependencies via rosdep
RUN apt-get update && apt-get install -y libzmq5-dev && \
    rosdep install --from-paths /tmp/deps/src --ignore-src -r -y --rosdistro=${ROS_DISTRO} || true && \
    apt remove ros-${ROS_DISTRO}-gtsam -y && \
    apt install ros-${ROS_DISTRO}-pcl-ros ros-${ROS_DISTRO}-diagnostic-updater libzip-dev ros-${ROS_DISTRO}-compressed-image-transport ros-${ROS_DISTRO}-compressed-depth-image-transport -y && \
    rm -rf /var/lib/apt/lists/*

# Open-set venv
RUN python3 -m venv --system-site-packages semantics_env && \
    /developer/semantics_env/bin/pip install --upgrade "pip<25" && \
    /developer/semantics_env/bin/pip install "setuptools<80" wheel && \
    /developer/semantics_env/bin/pip install "pybind11<2.12" && \
    cd /tmp/deps/src/spark_dsg && \
    /developer/semantics_env/bin/pip install . && \
    /developer/semantics_env/bin/pip install --no-build-isolation --no-deps \
        git+https://github.com/openai/CLIP.git && \
    /developer/semantics_env/bin/pip install -r /tmp/deps/src/semantic_inference/semantic_inference_python/requirements-thor.txt && \
    /developer/semantics_env/bin/pip install -r /tmp/deps/src/vlms_ros/vlms_python/requirements.txt && \
    /developer/semantics_env/bin/pip install ftfy && \
    rm -rf /tmp/deps/src
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:/opt/hpcx/ompi/lib:${LD_LIBRARY_PATH}
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

#useful shell configurations
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "alias qwe='source install/local_setup.bash'" >> ~/.bashrc && \
    echo "alias cb='colcon build --symlink-install --continue-on-error'" >> ~/.bashrc && \
    echo "export PS1='\[\e[1;32m\]\u@\h \[\e[1;34m\]\w\[\e[0m\]$ '" >> ~/.bashrc

CMD ["bash"]
