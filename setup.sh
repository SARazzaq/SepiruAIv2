#!/bin/bash
echo "🚀 Setting up CSV AI Analyst..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt
echo ""
echo "✅ Done! Next steps:"
echo "  1. Edit .env with your provider/API key"
echo "  2. streamlit run app.py"
