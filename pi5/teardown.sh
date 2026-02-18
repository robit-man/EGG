#!/bin/bash

set -e

TARGET_DIR="/home/$(whoami)/voice"
REPO_DIR="$TARGET_DIR/EGG_repo"
DOCKER_IMAGE_NAME="piper-tts-rpi5"
DOCKER_CONTAINER_NAME="piper-tts-pi5-service"

AUTOSTART_DIR="$HOME/.config/autostart"
AUTOSTART_FILES=(
    "$AUTOSTART_DIR/voice_setup.desktop"
    "$AUTOSTART_DIR/mini_egg_watchdog.desktop"
)

PYTHON_PROCESS_PATTERNS=(
    "$TARGET_DIR/whispercpp/examples/stream/stream.py"
    "$TARGET_DIR/output.py"
    "$TARGET_DIR/model_to_tts.py"
    "$TARGET_DIR/voice_server.py"
    "$TARGET_DIR/watchdog.py"
    "$TARGET_DIR/router.py"
    "$TARGET_DIR/camera_router.py"
    "$TARGET_DIR/pipeline_api.py"
    "$TARGET_DIR/audio_router.py"
    "$TARGET_DIR/run_asr_stream.py"
    "$TARGET_DIR/run_voice_server.py"
    "$TARGET_DIR/run_ollama_service.py"
)

OTHER_PROCESS_PATTERNS=(
    "$TARGET_DIR/nkn_sidecar/nkn_router_bridge.js"
)

FILES_TO_REMOVE=(
    "$TARGET_DIR/setup.sh"
    "$TARGET_DIR/teardown.sh"
    "$TARGET_DIR/sync_runtime.sh"
    "$TARGET_DIR/start_watchdog.sh"
    "$TARGET_DIR/watchdog.py"
    "$TARGET_DIR/router.py"
    "$TARGET_DIR/terminal_ui.py"
    "$TARGET_DIR/camera_router.py"
    "$TARGET_DIR/pipeline_api.py"
    "$TARGET_DIR/audio_router.py"
    "$TARGET_DIR/run_asr_stream.py"
    "$TARGET_DIR/run_voice_server.py"
    "$TARGET_DIR/run_ollama_service.py"
    "$TARGET_DIR/router_config.json"
    "$TARGET_DIR/camera_router_config.json"
    "$TARGET_DIR/pipeline_api_config.json"
    "$TARGET_DIR/audio_router_config.json"
    "$TARGET_DIR/camera_router_cloudflared"
    "$TARGET_DIR/pipeline_api_cloudflared"
    "$TARGET_DIR/audio_router_cloudflared"
    "$TARGET_DIR/output.py"
    "$TARGET_DIR/model_to_tts.py"
    "$TARGET_DIR/voice_server.py"
    "$TARGET_DIR/dockerfile"
    "$TARGET_DIR/glados_piper_medium.onnx"
    "$TARGET_DIR/glados_piper_medium.onnx.json"
    "$TARGET_DIR/chat.json"
    "$TARGET_DIR/config.json"
    "$TARGET_DIR/llm_bridge_config.json"
)

DIRS_TO_REMOVE=(
    "$TARGET_DIR/whispercpp"
    "$TARGET_DIR/watchdog_venv"
    "$TARGET_DIR/router_venv"
    "$TARGET_DIR/camera_router_venv"
    "$TARGET_DIR/pipeline_api_venv"
    "$TARGET_DIR/audio_router_venv"
    "$TARGET_DIR/venv"
    "$TARGET_DIR/.watchdog_runtime"
    "$TARGET_DIR/nkn_sidecar"
)

stop_process_patterns() {
    local patterns=("$@")
    for pattern in "${patterns[@]}"; do
        if pgrep -f "$pattern" > /dev/null 2>&1; then
            echo "[TEARDOWN] Stopping process pattern: $pattern"
            pkill -f "$pattern" || true
        fi
    done

    sleep 1

    for pattern in "${patterns[@]}"; do
        if pgrep -f "$pattern" > /dev/null 2>&1; then
            echo "[TEARDOWN] Force-killing process pattern: $pattern"
            pkill -9 -f "$pattern" || true
        fi
    done
}

stop_docker_stack() {
    if ! command -v docker > /dev/null 2>&1; then
        echo "[TEARDOWN] Docker is not installed. Skipping Docker cleanup."
        return
    fi

    local docker_cmd=()
    if docker info > /dev/null 2>&1; then
        docker_cmd=(docker)
    elif command -v sudo > /dev/null 2>&1 && sudo docker info > /dev/null 2>&1; then
        docker_cmd=(sudo docker)
    else
        echo "[TEARDOWN] Docker access unavailable; skipping Docker cleanup."
        return
    fi

    if "${docker_cmd[@]}" ps -a --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER_NAME}$"; then
        echo "[TEARDOWN] Removing container '${DOCKER_CONTAINER_NAME}'..."
        "${docker_cmd[@]}" rm -f "$DOCKER_CONTAINER_NAME" > /dev/null || true
    fi

    local containers
    containers=$("${docker_cmd[@]}" ps -aq --filter "ancestor=$DOCKER_IMAGE_NAME")
    if [ -n "$containers" ]; then
        echo "[TEARDOWN] Removing containers created from '${DOCKER_IMAGE_NAME}'..."
        # shellcheck disable=SC2086
        "${docker_cmd[@]}" rm -f $containers > /dev/null || true
    fi

    if "${docker_cmd[@]}" image inspect "$DOCKER_IMAGE_NAME" > /dev/null 2>&1; then
        echo "[TEARDOWN] Removing Docker image '$DOCKER_IMAGE_NAME'..."
        "${docker_cmd[@]}" rmi "$DOCKER_IMAGE_NAME" > /dev/null || true
    fi
}

remove_autostart_entry() {
    for file in "${AUTOSTART_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "[TEARDOWN] Removing autostart file: $file"
            rm -f "$file"
        fi
    done
}

remove_runtime_files() {
    for file in "${FILES_TO_REMOVE[@]}"; do
        if [ -e "$file" ]; then
            echo "[TEARDOWN] Removing file: $file"
            rm -f "$file"
        fi
    done

    for dir in "${DIRS_TO_REMOVE[@]}"; do
        if [ -d "$dir" ]; then
            echo "[TEARDOWN] Removing directory: $dir"
            rm -rf "$dir"
        fi
    done

    if [ -d "$REPO_DIR" ]; then
        echo "[TEARDOWN] Removing repository mirror: $REPO_DIR"
        rm -rf "$REPO_DIR"
    fi

    if [ -d "$TARGET_DIR" ] && [ -z "$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "[TEARDOWN] Removing empty directory: $TARGET_DIR"
        rmdir "$TARGET_DIR"
    fi
}

echo "[TEARDOWN] Starting Mini EGG teardown..."

stop_process_patterns "${PYTHON_PROCESS_PATTERNS[@]}"
stop_process_patterns "${OTHER_PROCESS_PATTERNS[@]}"
stop_docker_stack
remove_autostart_entry
remove_runtime_files

echo "[TEARDOWN] Teardown completed."
