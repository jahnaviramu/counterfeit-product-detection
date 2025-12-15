#!/usr/bin/env python
"""
Wrapper script to run Flask app with better error handling
"""
import sys
import os
import traceback

try:
    print("[INFO] Starting Flask backend server...")
    print("[INFO] Python version:", sys.version)
    print("[INFO] Working directory:", os.getcwd())
    
    # Import the app
    print("[INFO] Importing Flask app...")
    from app import app
    
    # Run the app
    print("[INFO] Starting Flask app on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    
except Exception as e:
    print(f"[FATAL ERROR] Failed to start server: {e}")
    traceback.print_exc()
    sys.exit(1)
