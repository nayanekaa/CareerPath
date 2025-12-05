#!/usr/bin/env python
"""Test if uvicorn works with a minimal app"""
from fastapi import FastAPI

app = FastAPI()

@app.get('/health')
def health():
    return {'status': 'ok'}

if __name__ == '__main__':
    import uvicorn
    print("Starting minimal fastapi server...")
    uvicorn.run(app, host='127.0.0.1', port=9000)
