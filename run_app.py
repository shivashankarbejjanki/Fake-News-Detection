#!/usr/bin/env python3
"""
Simple script to run the Fake News Detection web application
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'pandas', 'scikit-learn', 'nltk']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    return True

def main():
    """Main function to run the application"""
    print("🚀 Starting Fake News Detection System...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('streamlit_app.py'):
        print("❌ Error: streamlit_app.py not found!")
        print("Please run this script from the project directory.")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("✅ Dependencies check passed!")
    print("🌐 Starting web interface...")
    print("\nThe application will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.")
    print("=" * 50)
    
    try:
        # Run streamlit app
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running the application: {e}")
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
