#!/bin/bash
# Sequential alignment runner: German → Chinese → Spanish
# Survives SSH disconnection via nohup

echo "=========================================="
echo "🇩🇪 Starting GERMAN alignment..."
echo "=========================================="
cd /home/viswanath/llm-alignment/german && python3 llm_alignment_german.py
echo ""
echo "=========================================="
echo "🇨🇳 Starting CHINESE alignment..."
echo "=========================================="
cd /home/viswanath/llm-alignment/chinese && python3 llm_alignemnt_chinese.py
echo ""
echo "=========================================="
echo "🇪🇸 Starting SPANISH alignment..."
echo "=========================================="
cd /home/viswanath/llm-alignment/spanish && python3 llm_alignment_spanish.py
echo ""
echo "=========================================="
echo "🎉 ALL LANGUAGES COMPLETED!"
echo "=========================================="
