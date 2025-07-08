

OS := $(shell uname)

DOCKER_COMPOSE := docker compose

SERVICE_NAME := app

build:
ifeq ($(OS),Linux)
	@echo "Building for Linux..."
	$(DOCKER_COMPOSE) build
endif
ifeq ($(OS),Darwin)
	@echo "Building for macOS..."
	$(DOCKER_COMPOSE) build
endif
ifeq ($(OS),Windows_NT)
	@echo "Building for Windows..."
	$(DOCKER_COMPOSE) build
endif

run:
ifeq ($(OS),Linux)
	@echo "Running for Linux..."
	$(DOCKER_COMPOSE) up -d
endif
ifeq ($(OS),Darwin)
	@echo "Running for macOS..."
	$(DOCKER_COMPOSE) up -d
endif
ifeq ($(OS),Windows_NT)
	@echo "Running for Windows..."
	$(DOCKER_COMPOSE) up -d
endif

stop:
	$(DOCKER_COMPOSE) down

clean:
	@echo "Cleaning up..."
	$(DOCKER_COMPOSE) rm -f $(SERVICE_NAME)
	docker system prune -f