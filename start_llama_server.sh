#!/bin/bash

# Start llama server in the background
MODEL_PATH="models/LFM2-1.2B-RAG-Q4_K_M.gguf"
LOG_FILE="llama_server.log"
PID_FILE="llama_server.pid"

# Check if server is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Llama server is already running (PID: $PID)"
        exit 1
    else
        echo "Removing stale PID file"
        rm "$PID_FILE"
    fi
fi

# Start the server in the background
echo "Starting llama server..."
nohup llama-server -m "$MODEL_PATH" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Save the PID
echo "$SERVER_PID" > "$PID_FILE"

echo "Llama server started with PID: $SERVER_PID"
echo "Logs are being written to: $LOG_FILE"
echo "To stop the server, run: kill $SERVER_PID"
