#!/bin/bash
# Quick start script for dual_vllm

echo "================================"
echo "Dual VLLM Quick Start"
echo "================================"
echo ""

# Check if vllm is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "❌ vLLM not installed. Install with:"
    echo "   pip install vllm"
    exit 1
fi

# Default models
THETA_MODEL=${1:-"meta-llama/Llama-2-7b-hf"}
T_MODEL=${2:-"meta-llama/Llama-2-7b-chat-hf"}
THETA_PORT=${3:-8000}
T_PORT=${4:-8001}

echo "Configuration:"
echo "  π_θ (base):    $THETA_MODEL (port $THETA_PORT)"
echo "  π_t (teacher): $T_MODEL (port $T_PORT)"
echo ""

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0
    else
        return 1
    fi
}

# Check if ports are already in use
if check_port $THETA_PORT; then
    echo "⚠️  Port $THETA_PORT already in use (π_θ may already be running)"
else
    echo "Starting π_θ on port $THETA_PORT..."
    python -m vllm.entrypoints.api_server \
        --model "$THETA_MODEL" \
        --port $THETA_PORT \
        --dtype auto \
        > vllm_theta.log 2>&1 &
    THETA_PID=$!
    echo "  PID: $THETA_PID"
fi

if check_port $T_PORT; then
    echo "⚠️  Port $T_PORT already in use (π_t may already be running)"
else
    echo "Starting π_t on port $T_PORT..."
    python -m vllm.entrypoints.api_server \
        --model "$T_MODEL" \
        --port $T_PORT \
        --dtype auto \
        > vllm_t.log 2>&1 &
    T_PID=$!
    echo "  PID: $T_PID"
fi

echo ""
echo "Waiting for vLLM instances to be ready..."
sleep 5

# Wait for both instances to be ready
wait_for_vllm() {
    local url=$1
    local max_wait=120
    local waited=0

    while [ $waited -lt $max_wait ]; do
        if curl -s "$url/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done
    echo ""
    return 1
}

echo -n "Checking π_θ (http://localhost:$THETA_PORT)"
if wait_for_vllm "http://localhost:$THETA_PORT"; then
    echo " ✅"
else
    echo " ❌ Failed to start"
    exit 1
fi

echo -n "Checking π_t (http://localhost:$T_PORT)"
if wait_for_vllm "http://localhost:$T_PORT"; then
    echo " ✅"
else
    echo " ❌ Failed to start"
    exit 1
fi

echo ""
echo "================================"
echo "✅ Both vLLM instances are ready!"
echo "================================"
echo ""
echo "You can now run:"
echo "  python example.py"
echo ""
echo "To stop the instances:"
echo "  kill $THETA_PID $T_PID"
echo ""
echo "View logs:"
echo "  tail -f vllm_theta.log"
echo "  tail -f vllm_t.log"
