# ShowdownAI

## Docker

Manage `libwebsockets` via Docker to ensure a consistent build environment across platforms.

### Prerequisites
- Install Docker Desktop.

### Build
- Run: `docker build -t project-porygon .`

### Run
- Run: `docker run --rm -it project-porygon`

This uses an Ubuntu base image, installs `libwebsockets-dev`, configures with CMake (Ninja generator), and runs the compiled `showdown_client` binary from `build/`.