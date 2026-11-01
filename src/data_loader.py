"""
Fitness and Workout Progress Analyzer
Module: data_loader.py
Description: Data loading and preprocessing pipelines for gym exercise tracking,
             calories burned, and exercise metadata.
"""

import os
import json
import pandas as pd
import numpy as np

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

GYM_DATASET_PATH = os.path.join(DATASETS_DIR, "datasets/gym_members_exercise_tracking.csv")
CALORIES_DATASET_PATH = os.path.join(DATASETS_DIR, "datasets/calories.csv")
EXERCISES_DATASET_PATH = os.path.join(
    DATASETS_DIR, "datasets/exercises-dataset-main/data/exercises.json"
)

def _resolve_dataset_path(path: str, filename: str) -> str:
    """Helper to locate dataset in datasets/ or root."""
    if os.path.exists(path):
        return path
    fallback_datasets = os.path.join(DATASETS_DIR, filename)
    if os.path.exists(fallback_datasets):
        return fallback_datasets
    fallback_root = os.path.join(PROJECT_ROOT, filename)
    if os.path.exists(fallback_root):
        return fallback_root
    return path

def load_gym_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Load the primary gym members exercise tracking dataset.
    Contains workout sessions, biometrics, workout types, and calories burned.
    """
    path = _resolve_dataset_path(filepath or GYM_DATASET_PATH, "gym_members_exercise_tracking.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gym dataset not found at {path}")
    
    df = pd.read_csv(path)
    # Strip any leading/trailing whitespace in column names
    df.columns = [col.strip() for col in df.columns]
    
    # Calculate BMI if missing or inconsistent
    if "BMI" not in df.columns and "Weight (kg)" in df.columns and "Height (m)" in df.columns:
        df["BMI"] = np.round(df["Weight (kg)"] / (df["Height (m)"] ** 2), 2)
        
    return df

def load_calories_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Load the secondary large-scale calories dataset (15,000 observations).
    Useful for cross-validation, hypothesis testing, and population sampling.
    """
    path = _resolve_dataset_path(filepath or CALORIES_DATASET_PATH, "calories.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calories dataset not found at {path}")
    
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    return df

def load_exercises_dataset(filepath: str = None) -> list:
    """
    Load the comprehensive exercise library (1,324 exercises).
    Contains body parts, target muscles, equipments, instructions, and categories.
    """
    target_rel = os.path.join("exercises-dataset-main", "data", "exercises.json")
    path = filepath or EXERCISES_DATASET_PATH
    if not os.path.exists(path):
        fallback_datasets = os.path.join(DATASETS_DIR, target_rel)
        if os.path.exists(fallback_datasets):
            path = fallback_datasets
        else:
            fallback_root = os.path.join(PROJECT_ROOT, target_rel)
            if os.path.exists(fallback_root):
                path = fallback_root
            else:
                raise FileNotFoundError(f"Exercises JSON not found at {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        exercises = json.load(f)
    return exercises

def get_ml_feature_names():
    """
    Returns feature specifications for the Calorie Prediction ML models.
    """
    numeric_features = [
        "Age",
        "Weight (kg)",
        "Height (m)",
        "Max_BPM",
        "Avg_BPM",
        "Resting_BPM",
        "Session_Duration (hours)",
        "Fat_Percentage",
        "Water_Intake (liters)",
        "Workout_Frequency (days/week)",
        "Experience_Level",
        "BMI"
    ]
    categorical_features = ["Gender", "Workout_Type"]
    target = "Calories_Burned"
    return numeric_features, categorical_features, target

if __name__ == "__main__":
    gym = load_gym_dataset()
    print("Gym Members Dataset Loaded successfully. Shape:", gym.shape)
    cal = load_calories_dataset()
    print("Calories Dataset Loaded successfully. Shape:", cal.shape)
    ex = load_exercises_dataset()
    print("Exercises Dataset Loaded successfully. Count:", len(ex))

