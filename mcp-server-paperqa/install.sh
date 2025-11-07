#!/bin/bash
# Installation script for PaperQA MCP Server

set -e

echo "==================================="
echo "PaperQA MCP Server Installation"
echo "==================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.11+ is required (found $python_version)"
    exit 1
fi
echo "✓ Python $python_version"
echo ""

# Install the server
echo "Installing PaperQA MCP Server..."
pip install -e .
echo "✓ Server installed"
echo ""

# Check if Claude Code config directory exists
config_dir="$HOME/.config/claude"
if [ ! -d "$config_dir" ]; then
    echo "Creating Claude Code config directory..."
    mkdir -p "$config_dir"
fi

# Check if mcp_config.json exists
config_file="$config_dir/mcp_config.json"
if [ -f "$config_file" ]; then
    echo "⚠ Warning: $config_file already exists"
    echo "Please manually add the PaperQA server configuration from mcp_config.example.json"
else
    echo "Creating default MCP configuration..."

    # Prompt for paper directory
    read -p "Enter the path to your papers directory (or press Enter for ~/papers): " paper_dir
    paper_dir=${paper_dir:-"$HOME/papers"}

    # Create papers directory if it doesn't exist
    if [ ! -d "$paper_dir" ]; then
        echo "Creating papers directory at $paper_dir..."
        mkdir -p "$paper_dir"
    fi

    # Prompt for OpenAI API key
    read -p "Enter your OpenAI API key (or press Enter to skip): " openai_key

    # Create config
    cat > "$config_file" <<EOF
{
  "mcpServers": {
    "paperqa": {
      "command": "paperqa-mcp",
      "env": {
        "PAPERQA_PAPER_DIRECTORY": "$paper_dir",
        "PAPERQA_INDEX_DIRECTORY": "\${HOME}/.paperqa/indexes",
        "PAPERQA_SETTINGS": "fast"
EOF

    if [ -n "$openai_key" ]; then
        cat >> "$config_file" <<EOF
,
        "OPENAI_API_KEY": "$openai_key"
EOF
    fi

    cat >> "$config_file" <<EOF

      }
    }
  }
}
EOF

    echo "✓ Configuration created at $config_file"
fi

echo ""
echo "==================================="
echo "Installation Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Make sure you have papers in: $paper_dir"
echo "2. Set your OPENAI_API_KEY if not already done"
echo "3. Restart Claude Code"
echo "4. Use the PaperQA tools in Claude Code!"
echo ""
echo "Example usage:"
echo '  "Claude, use paperqa_ask to answer: What are the latest advances in AI?"'
echo ""
