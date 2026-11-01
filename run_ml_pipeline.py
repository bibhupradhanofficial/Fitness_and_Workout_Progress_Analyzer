"""
Fitness and Workout Progress Analyzer
Script: run_ml_pipeline.py
Description: Master CLI runner to train, cross-validate, evaluate, and save all ML models.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.data_loader import load_gym_dataset
from src.ml_models import train_and_evaluate_models

def main():
    print("\nStarting Machine Learning Training Pipeline for Calorie Burn Prediction...")
    df = load_gym_dataset()
    results = train_and_evaluate_models(df)
    print("\nML Pipeline successfully completed! Trained models saved in 'models/' and plots in 'output_plots/'.")

if __name__ == "__main__":
    main()

