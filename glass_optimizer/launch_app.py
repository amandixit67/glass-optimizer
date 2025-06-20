#!/usr/bin/env python3
"""
Launcher script for AI Glass Optimization Tool
Run this from any directory to start the application
"""

import os
import sys
import subprocess

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the correct directory
    os.chdir(script_dir)
    
    print("🔷 Starting AI Glass Optimization Tool...")
    print(f"📁 Working directory: {script_dir}")
    print("🌐 Opening browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Run the Streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("💡 Make sure you have installed the requirements:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 