#!/bin/bash
set -e

REPO_DIR="${1:-/home/$(whoami)/voice/EGG_repo}"
TARGET_DIR="${2:-/home/$(whoami)/voice}"

if [ ! -d "$REPO_DIR" ]; then
    echo "[SYNC] Repository directory not found: $REPO_DIR"
    exit 1
fi

mkdir -p "$TARGET_DIR"

copy_file() {
    local src="$1"
    local dst="$2"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        chmod +x "$dst"
    else
        echo "[SYNC] Missing source file: $src"
    fi
}

# Whisper runtime
if [ -d "$REPO_DIR/pi5/whisper/whispercpp" ]; then
    if command -v rsync > /dev/null 2>&1; then
        rsync -a --delete "$REPO_DIR/pi5/whisper/whispercpp/" "$TARGET_DIR/whispercpp/"
    else
        rm -rf "$TARGET_DIR/whispercpp"
        cp -r "$REPO_DIR/pi5/whisper/whispercpp" "$TARGET_DIR/"
    fi
fi

# Piper scripts and dockerfile
copy_file "$REPO_DIR/pi5/piper/output.py" "$TARGET_DIR/output.py"
copy_file "$REPO_DIR/pi5/piper/model_to_tts.py" "$TARGET_DIR/model_to_tts.py"
copy_file "$REPO_DIR/pi5/piper/voice_server.py" "$TARGET_DIR/voice_server.py"
copy_file "$REPO_DIR/pi5/piper/dockerfile" "$TARGET_DIR/dockerfile"

# Mini EGG teleop/watchdog stack
copy_file "$REPO_DIR/pi5/setup.sh" "$TARGET_DIR/setup.sh"
copy_file "$REPO_DIR/pi5/teardown.sh" "$TARGET_DIR/teardown.sh"
copy_file "$REPO_DIR/pi5/sync_runtime.sh" "$TARGET_DIR/sync_runtime.sh"
copy_file "$REPO_DIR/pi5/watchdog.py" "$TARGET_DIR/watchdog.py"
copy_file "$REPO_DIR/pi5/router.py" "$TARGET_DIR/router.py"
copy_file "$REPO_DIR/pi5/terminal_ui.py" "$TARGET_DIR/terminal_ui.py"
copy_file "$REPO_DIR/pi5/camera_router.py" "$TARGET_DIR/camera_router.py"
copy_file "$REPO_DIR/pi5/pipeline_api.py" "$TARGET_DIR/pipeline_api.py"
copy_file "$REPO_DIR/pi5/audio_router.py" "$TARGET_DIR/audio_router.py"
copy_file "$REPO_DIR/pi5/run_asr_stream.py" "$TARGET_DIR/run_asr_stream.py"
copy_file "$REPO_DIR/pi5/run_voice_server.py" "$TARGET_DIR/run_voice_server.py"
copy_file "$REPO_DIR/pi5/run_ollama_service.py" "$TARGET_DIR/run_ollama_service.py"

echo "[SYNC] Runtime files synchronized to $TARGET_DIR"
