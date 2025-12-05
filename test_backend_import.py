#!/usr/bin/env python
"""Minimal test of backend imports and startup"""
import sys
sys.path.insert(0, r'c:\Users\HP\Downloads\careerpath-ai')

try:
    print("Testing backend.main import...")
    from backend import main
    print("✓ backend.main imported")
    print(f"✓ FastAPI app: {main.app}")
except Exception as e:
    print(f"✗ Error importing backend.main: {e}")
    import traceback
    traceback.print_exc()
