#!/usr/bin/env python
"""Direct test of uvicorn with backend"""
import sys
sys.path.insert(0, r'c:\Users\HP\Downloads\careerpath-ai')

if __name__ == '__main__':
    import uvicorn
    from backend.main import app
    print("Starting uvicorn server on 127.0.0.1:8002...")
    uvicorn.run(app, host='127.0.0.1', port=8002, log_level='debug')
