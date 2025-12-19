#!/bin/bash

echo "🚀 Setting up Vectify environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create directory structure
python config.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Place raw SVG files in data_raw/svgs_raw/"
echo "  2. Run: python scripts/clean_svg.py"
