#!/bin/bash
cd "$(dirname "$0")"
echo ""
echo "  Insurance Claims Fraud ML"
echo "============================================"
echo "  Copy this link and paste it in any browser"
echo "  or on another device:"
echo ""
echo "  http://127.0.0.1:5000"
echo ""
echo "============================================"
echo ""
PY=$(command -v python3 || command -v python)
exec $PY app.py
