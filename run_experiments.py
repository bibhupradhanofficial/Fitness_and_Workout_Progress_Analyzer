"""
Fitness and Workout Progress Analyzer
Script: run_experiments.py
Description: Master CLI runner to execute all 8 DAP Lab Experiments and produce artifacts.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.data_loader import load_gym_dataset
from src.dap_lab_experiments import run_all_experiments

def main():
    print("\nStarting execution of all 8 Data Analysis with Python Lab Experiments...")
    df = load_gym_dataset()
    results = run_all_experiments(df)
    print("\nAll experiments completed successfully! Visualizations saved in 'output_plots/'.")

if __name__ == "__main__":
    main()

