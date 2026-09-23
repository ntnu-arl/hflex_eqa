# Base image with CUDA 12.8 and Ubuntu 24.04
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy

#setup ros2 workspace (mounted from host)
WORKDIR /developer

#ros2 jazzy
RUN apt-get update && \
    apt-get install -y curl gnupg2 lsb-release software-properties-common && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /etc/apt/trusted.gpg.d/ros.gpg && \
    echo "deb http://packages.ros.org/ros2/ubuntu noble main" > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && \
    apt-get install -y ros-${ROS_DISTRO}-desktop python3-rosdep python3-colcon-common-extensions && \
    apt-get install -y ros-${ROS_DISTRO}-depth-image-proc ros-${ROS_DISTRO}-rqt-tf-tree && \
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
    ros-dev-tools \
    nlohmann-json3-dev \
    libgoogle-glog-dev \
    libcli11-dev \
    libeigen3-dev \
    libnanoflann-dev && \
    rm -rf /var/lib/apt/lists/*

#install pipx + rosbags
RUN apt-get update && apt-get install -y pipx && \
    pipx install rosbags && \
    echo 'export PATH=$PATH:/root/.local/bin' >> /root/.bashrc && \
    rm -rf /var/lib/apt/lists/*

#install tensorrt (must remove conflicting cuda repo to make work, unsure why
#cuda12.8 comes with the base image instead of cuda12.6)
RUN rm -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list && \
    rm -f /etc/apt/sources.list.d/cuda*.list && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y \
        libnvinfer-dev=10.8.0.43-1+cuda12.8 \
        libnvinfer10=10.8.0.43-1+cuda12.8 \
        libnvinfer-headers-dev=10.8.0.43-1+cuda12.8 \
        libnvinfer-plugin-dev=10.8.0.43-1+cuda12.8 \
        libnvinfer-plugin10=10.8.0.43-1+cuda12.8 \
        libnvinfer-headers-plugin-dev=10.8.0.43-1+cuda12.8 \
        libnvonnxparsers-dev=10.8.0.43-1+cuda12.8 \
        libnvonnxparsers10=10.8.0.43-1+cuda12.8 && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.1-1_all.deb

#networking
RUN apt update && apt install -y ros-jazzy-rmw-zenoh-cpp \
    python3-colcon-clean libdxflib-dev

# Install Open3D
RUN git clone --branch v0.19.0 https://github.com/isl-org/Open3D.git --depth 1 && cd Open3D && \
    util/install_deps_ubuntu.sh assume-yes && \
    mkdir build && cd build && cmake .. && make -j$(nproc) && make install && \
    mkdir -p /developer/share/Open3D && \
    ln -s /usr/local/bin/Open3D/resources /developer/share/Open3D/resources

# Copy source code for dependency discovery (replaced by a volume at runtime).
COPY src/ /tmp/deps/src/

#install dependencies via rosdep
RUN apt-get update && apt-get install -y libzmq5-dev ros-${ROS_DISTRO}-gtsam ros-${ROS_DISTRO}-rmw-cyclonedds-cpp  && \
    rosdep install --from-paths /tmp/deps/src --ignore-src -r -y --rosdistro=${ROS_DISTRO} || true

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

# Open-set venv
RUN python3 -m venv --system-site-packages semantics_env && \
    /developer/semantics_env/bin/pip install --upgrade "pip<25" && \
    /developer/semantics_env/bin/pip install "setuptools<80" wheel && \
    /developer/semantics_env/bin/pip install "pybind11<2.12" && \
    cd /tmp/deps/src/spark_dsg && \
    /developer/semantics_env/bin/pip install . && \
    /developer/semantics_env/bin/pip install --no-build-isolation --no-deps \
        git+https://github.com/openai/CLIP.git && \
    /developer/semantics_env/bin/pip install -r /tmp/deps/src/semantic_inference/semantic_inference_python/requirements.txt && \
    /developer/semantics_env/bin/pip install -r /tmp/deps/src/vlms_ros/vlms_python/requirements.txt && \
    /developer/semantics_env/bin/pip install ftfy

# Habitat-sim venv
ENV CC=gcc-11
ENV CXX=g++-11
RUN python3 -m venv --system-site-packages habitat_env && \
    /developer/habitat_env/bin/pip install --upgrade "pip<25" "setuptools<80" wheel && \
    /developer/habitat_env/bin/pip install -r /tmp/deps/src/habitat-ros/requirements.txt && \
    cd /tmp/deps/src/habitat-ros/habitat_sim && \
    rm -rf build && \
    /developer/habitat_env/bin/python setup.py install --headless --with-cuda --cmake-args="-DUSE_SYSTEM_ASSIMP=ON" && \
    rm -rf /tmp/deps/src
ENV CC=
ENV CXX=
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

#useful shell configurations
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "alias qwe='source install/local_setup.bash'" >> ~/.bashrc && \
    echo "alias cb='colcon build --symlink-install --continue-on-error'" >> ~/.bashrc && \
    echo "export PS1='\[\e[1;32m\]\u@\h \[\e[1;34m\]\w\[\e[0m\]$ '" >> ~/.bashrc

CMD ["bash"]
