#!/usr/bin/env python3
"""
Healthcare Queue Optimizer - One-Click Launcher
================================================

This script automatically sets up and runs the Healthcare Queue Optimizer.
It will:
1. Install required dependencies
2. Generate synthetic training data (if needed)
3. Train the AI model (if needed)
4. Launch the Streamlit web application

Usage:
    python run.py
    
or simply:
    python run.py --port 8501
"""

import os
import sys
import subprocess
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit
        import torch
        import transformers
        import pandas
        import numpy
        import matplotlib
        import joblib
        import sklearn
        return True
    except ImportError as e:
        logger.warning(f"Missing dependency: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_data():
    """Generate synthetic data if needed"""
    data_path = "data/synthetic_data.csv"
    if not os.path.exists(data_path):
        logger.info("Generating synthetic training data...")
        try:
            subprocess.check_call([sys.executable, "synthetic_data.py"])
            logger.info("Synthetic data generated successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate synthetic data: {e}")
            return False
    else:
        logger.info("Synthetic data already exists.")
        return True

def setup_model():
    """Train model if needed"""
    model_path = "model/healthcare_model.pt"
    if not os.path.exists(model_path):
        logger.info("Training AI model (this may take a few minutes)...")
        try:
            subprocess.check_call([sys.executable, "-m", "src.train"])
            logger.info("Model training completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to train model: {e}")
            return False
    else:
        logger.info("Trained model already exists.")
        return True

def launch_app(port=8501):
    """Launch the Streamlit application"""
    logger.info(f"Launching Healthcare Queue Optimizer on port {port}...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port)])
    except KeyboardInterrupt:
        logger.info("Application stopped by user.")
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")

def main():
    parser = argparse.ArgumentParser(description="Healthcare Queue Optimizer Launcher")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on (default: 8501)")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining")
    
    args = parser.parse_args()
    
    print("🏥 Healthcare Emergency Queue Optimizer")
    print("=" * 50)
    
    # Check and install dependencies
    if not args.skip_deps:
        if not check_dependencies():
            print("📦 Installing missing dependencies...")
            if not install_dependencies():
                print("❌ Failed to install dependencies. Please install manually.")
                return 1
    
    # Setup data
    print("📊 Setting up training data...")
    if not setup_data():
        print("❌ Failed to setup training data.")
        return 1
    
    # Setup model
    if args.force_retrain and os.path.exists("model/healthcare_model.pt"):
        os.remove("model/healthcare_model.pt")
        print("🔄 Forcing model retraining...")
    
    print("🤖 Setting up AI model...")
    if not setup_model():
        print("❌ Failed to setup AI model.")
        return 1
    
    # Launch application
    print("🚀 Setup complete! Launching application...")
    print(f"📱 Open your browser to: http://localhost:{args.port}")
    print("Press Ctrl+C to stop the application")
    
    launch_app(args.port)
    return 0

if __name__ == "__main__":
    sys.exit(main())
