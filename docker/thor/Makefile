COMPOSE ?= docker compose
COMPOSE_FILE ?= docker-compose.yaml
SERVICE ?= ros2_hflex_eqa
CONTAINER ?= hflex_eqa_thor
HFLEX_WS ?= $(abspath ../../../..)
RESULTS_DIR ?= $(HFLEX_WS)/results
BAGS_DIR ?= $(HFLEX_WS)/bags
export HFLEX_WS RESULTS_DIR BAGS_DIR

.PHONY: build run launch stop attach logs

build:
	$(COMPOSE) -f $(COMPOSE_FILE) build $(SERVICE)

run:
	$(COMPOSE) -f $(COMPOSE_FILE) run --rm --name $(CONTAINER) $(SERVICE)

launch:
	$(COMPOSE) -f $(COMPOSE_FILE) up scene_graph eqa

stop:
	$(COMPOSE) -f $(COMPOSE_FILE) down

attach:
	docker exec -it $(CONTAINER) /bin/bash

logs:
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f scene_graph eqa
