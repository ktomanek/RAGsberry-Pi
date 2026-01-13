#!/bin/bash

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <model_path>"
    echo "Example: $0 models/LFM2-1.2B-RAG-Q4_K_M.gguf"
    exit 1
fi

MODEL_PATH="$1"
LOG_FILE="llama_server.log"
PID_FILE="llama_server.pid"

# Delete old log and PID files if they exist
[ -f "$LOG_FILE" ] && rm "$LOG_FILE"
[ -f "$PID_FILE" ] && rm "$PID_FILE"

# Start the server in the background
echo "Starting llama server with model: $MODEL_PATH"
nohup llama-server -m "$MODEL_PATH" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Save the PID
echo "$SERVER_PID" > "$PID_FILE"

echo "Llama server started with PID: $SERVER_PID"
echo "Logs are being written to: $LOG_FILE"
echo "To stop the server, run: kill $SERVER_PID"
