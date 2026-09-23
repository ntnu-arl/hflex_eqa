COMPOSE ?= docker compose
COMPOSE_FILE ?= docker-compose.yaml
SERVICE ?= ros2_hvlm
LAUNCH_SERVICE ?= ros2_hvlm_launch
CONTAINER ?= hflex_eqa
HFLEX_WS ?= $(abspath ../../../..)
HM3D_DATA ?= $(HFLEX_WS)/data/hm3d
RESULTS_DIR ?= $(HFLEX_WS)/results
export HFLEX_WS HM3D_DATA RESULTS_DIR

.PHONY: build run simulation dataset stop attach logs

build:
	$(COMPOSE) -f $(COMPOSE_FILE) build $(SERVICE)

run:
	$(COMPOSE) -f $(COMPOSE_FILE) run --rm --name $(CONTAINER) $(SERVICE)

simulation:
	@test -n "$(SCENE_FILE)" || (echo "Set SCENE_FILE to a .glb path inside the container" && exit 1)
	$(COMPOSE) -f $(COMPOSE_FILE) run --rm $(SERVICE) bash -lc \
		'cd /developer/hflex_eqa_ws && source install/setup.bash && ros2 launch hvlm_planner_ros habitat_eqa.launch.yaml scene_file:=$(SCENE_FILE)'

dataset:
	$(COMPOSE) -f $(COMPOSE_FILE) up $(LAUNCH_SERVICE)

stop:
	$(COMPOSE) -f $(COMPOSE_FILE) down

attach:
	docker exec -it $(CONTAINER) /bin/bash

logs:
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f $(LAUNCH_SERVICE)
