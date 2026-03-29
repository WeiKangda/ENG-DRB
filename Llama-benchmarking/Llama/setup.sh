#!/bin/bash

# Llama Benchmarking Pipeline Setup Script

echo "Setting up Llama Benchmarking Pipeline..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $python_version is installed, but Python $required_version or higher is required."
    exit 1
fi

echo "Python version: $python_version ✓"

# Create virtual environment (optional)
read -p "Do you want to create a virtual environment? (y/n): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv llama_env
    echo "Virtual environment created. Activate it with:"
    echo "source llama_env/bin/activate  # On macOS/Linux"
    echo "llama_env\\Scripts\\activate     # On Windows"
    echo ""
    echo "After activating the virtual environment, run:"
    echo "pip install -r requirements.txt"
else
    echo "Installing dependencies in current environment..."
    pip3 install -r requirements.txt
fi

# Create system prompt
echo "Creating default system prompt..."
python3 llama_benchmarking_pipeline.py --create-prompt

# Check if data files exist
if [ ! -f "../revised_data.jsonl" ]; then
    echo "Warning: ../revised_data.jsonl not found."
    echo "Please ensure your data files are in the correct location."
else
    echo "Data file found: ../revised_data.jsonl ✓"
fi

# Create results directory
mkdir -p results

echo ""
echo "Setup completed!"
echo ""
echo "Next steps:"
echo "1. Request access to Llama models on Hugging Face:"
echo "   https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
echo ""
echo "2. Login to Hugging Face:"
echo "   huggingface-cli login"
echo ""
echo "3. Run the pipeline:"
echo "   python3 llama_benchmarking_pipeline.py"
echo ""
echo "4. For help with options:"
echo "   python3 llama_benchmarking_pipeline.py --help" 