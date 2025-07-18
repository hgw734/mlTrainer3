#!/bin/bash
exec streamlit run mltrainer_unified_chat.py --server.port="${PORT:-8501}" --server.address=0.0.0.0
