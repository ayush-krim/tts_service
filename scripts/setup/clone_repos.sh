#!/bin/bash
# clone_repos.sh - Clone required repositories for Chatterbox optimization

set -e

echo "=================================================="
echo "CLONING CHATTERBOX REPOSITORIES"
echo "=================================================="

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Project root: $PROJECT_ROOT"

# Create repos directory
mkdir -p "$PROJECT_ROOT/repos"
cd "$PROJECT_ROOT/repos"

echo ""
echo "1. Cloning official Chatterbox..."
echo "----------------------------------"
if [ -d "chatterbox" ]; then
    echo "   Already exists, pulling latest..."
    cd chatterbox && git pull && cd ..
else
    git clone https://github.com/resemble-ai/chatterbox.git
fi

echo ""
echo "2. Cloning chatterbox-streaming..."
echo "-----------------------------------"
if [ -d "chatterbox-streaming" ]; then
    echo "   Already exists, pulling latest..."
    cd chatterbox-streaming && git pull && cd ..
else
    git clone https://github.com/davidbrowne17/chatterbox-streaming.git
fi

echo ""
echo "3. Cloning chatterbox-vllm..."
echo "------------------------------"
if [ -d "chatterbox-vllm" ]; then
    echo "   Already exists, pulling latest..."
    cd chatterbox-vllm && git pull && cd ..
else
    git clone https://github.com/randombk/chatterbox-vllm.git
fi

echo ""
echo "=================================================="
echo "REPOSITORIES CLONED"
echo "=================================================="
echo ""
ls -la "$PROJECT_ROOT/repos/"
