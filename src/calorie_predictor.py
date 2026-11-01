"""
Fitness and Workout Progress Analyzer
Module: calorie_predictor.py
Description: Dual-tier calorie prediction engine:
             1. Session-level Machine Learning Prediction (Trained Pipeline)
             2. Exercise-specific Set-Rep-Weight-Muscle Group Caloric Engine
"""

import os
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_bundle.joblib")

# Exercise MET & Muscle Mass Recruitment Coefficients
# Compound multi-joint exercises recruit larger muscle masses and expend more energy.
MUSCLE_GROUP_METS = {
    "upper legs": {"base_met": 7.5, "compound_factor": 1.25, "rom_m": 0.65},  # Squats, Lunges, Leg Press
    "back": {"base_met": 7.0, "compound_factor": 1.20, "rom_m": 0.55},        # Deadlifts, Rows, Pull-ups
    "chest": {"base_met": 6.5, "compound_factor": 1.15, "rom_m": 0.45},       # Bench Press, Push-ups
    "shoulders": {"base_met": 6.0, "compound_factor": 1.10, "rom_m": 0.45},   # Overhead Press, Lateral Raises
    "upper arms": {"base_met": 5.0, "compound_factor": 1.00, "rom_m": 0.35},  # Curls, Tricep Extensions
    "lower arms": {"base_met": 4.5, "compound_factor": 1.00, "rom_m": 0.25},  # Wrist Curls, Forearms
    "waist": {"base_met": 5.5, "compound_factor": 1.05, "rom_m": 0.30},       # Planks, Crunches, Leg Raises
    "cardio": {"base_met": 8.5, "compound_factor": 1.30, "rom_m": 0.50},      # Burpees, Mountain Climbers
    "general": {"base_met": 5.5, "compound_factor": 1.05, "rom_m": 0.40}
}

class CaloriePredictor:
    def __init__(self, model_bundle_path: str = BEST_MODEL_PATH):
        self.model_bundle_path = model_bundle_path
        self.model_bundle = None
        self.pipeline = None
        self._load_model()
        
    def _load_model(self):
        """Loads the pre-trained ML model bundle if available."""
        if os.path.exists(self.model_bundle_path):
            try:
                self.model_bundle = joblib.load(self.model_bundle_path)
                self.pipeline = self.model_bundle["pipeline"]
            except Exception as e:
                print(f"Warning: Could not load model from {self.model_bundle_path}: {e}")
                self.pipeline = None
        else:
            self.pipeline = None

    def predict_session_calories(
        self,
        age: int = 28,
        gender: str = "Male",
        weight_kg: float = 75.0,
        height_m: float = 1.75,
        session_duration_hours: float = 1.0,
        workout_type: str = "Strength",
        avg_bpm: int = 145,
        max_bpm: int = 175,
        resting_bpm: int = 62,
        fat_percentage: float = 18.0,
        water_intake_liters: float = 2.5,
        workout_frequency_days: int = 4,
        experience_level: int = 2
    ) -> dict:
        """
        Predicts total session calorie expenditure using the trained Machine Learning pipeline.
        """
        bmi = round(weight_kg / (height_m ** 2), 2)
        
        input_dict = {
            "Age": [age],
            "Gender": [gender],
            "Weight (kg)": [weight_kg],
            "Height (m)": [height_m],
            "Max_BPM": [max_bpm],
            "Avg_BPM": [avg_bpm],
            "Resting_BPM": [resting_bpm],
            "Session_Duration (hours)": [session_duration_hours],
            "Workout_Type": [workout_type],
            "Fat_Percentage": [fat_percentage],
            "Water_Intake (liters)": [water_intake_liters],
            "Workout_Frequency (days/week)": [workout_frequency_days],
            "Experience_Level": [experience_level],
            "BMI": [bmi]
        }
        input_df = pd.DataFrame(input_dict)
        
        if self.pipeline is not None:
            predicted_cal = float(self.pipeline.predict(input_df)[0])
            model_used = self.model_bundle.get("model_name", "Trained ML Model")
        else:
            # Fallback physics calculation if model bundle is not yet loaded
            intensity_factor = 1.2 if workout_type in ["HIIT", "Cardio"] else 1.0
            predicted_cal = (avg_bpm * 0.2 + weight_kg * 0.08) * (session_duration_hours * 60) * 0.14 * intensity_factor
            model_used = "Biophysical Fallback Estimator"
            
        return {
            "predicted_calories": round(predicted_cal, 1),
            "model_architecture": model_used,
            "session_duration_hours": session_duration_hours,
            "workout_type": workout_type,
            "bmi": bmi,
            "burn_rate_kcal_per_hour": round(predicted_cal / max(session_duration_hours, 0.1), 1)
        }

    def predict_exercise_calories(
        self,
        exercise_name: str,
        muscle_group: str,
        sets: int,
        reps: int,
        weight_lifted_kg: float,
        user_body_weight_kg: float = 75.0,
        rest_seconds_between_sets: int = 60
    ) -> dict:
        """
        Calculates calories burned for a specific resistance exercise based on:
        - Exercise name and muscle group
        - Sets, Reps, and Weight lifted (kg)
        - Mechanical work (W = F * d = m * g * h)
        - Muscle efficiency (~20%)
        - Metabolic recovery during rest intervals (EPOC & baseline MET)
        """
        muscle_key = muscle_group.lower().strip()
        profile = MUSCLE_GROUP_METS.get(muscle_key, MUSCLE_GROUP_METS["general"])
        
        # 1. Active time per rep ~ 3.5 seconds (tempo: 2s eccentric, 1s concentric, 0.5s pause)
        time_per_rep_sec = 3.5
        active_time_per_set_sec = reps * time_per_rep_sec
        total_active_time_min = (sets * active_time_per_set_sec) / 60.0
        total_rest_time_min = ((sets - 1) * rest_seconds_between_sets) / 60.0
        total_exercise_duration_min = total_active_time_min + total_rest_time_min
        
        # 2. Mechanical Work Done
        # Mass moved = external weight + fraction of body mass involved
        body_mass_fraction = 0.65 if muscle_key in ["upper legs", "waist"] else 0.15
        effective_mass = weight_lifted_kg + (user_body_weight_kg * body_mass_fraction)
        
        # Range of motion
        rom = profile["rom_m"]
        g = 9.81  # m/s^2
        total_reps = sets * reps
        
        # Work in Joules = Force * Distance = (m * g) * (rom * reps)
        work_joules = effective_mass * g * rom * total_reps
        
        # Convert Joules to kcal: 1 kcal = 4184 J
        # Skeletal muscle thermodynamic efficiency is ~20% (0.20)
        # Metabolic energy expended = Work / Efficiency
        metabolic_work_kcal = (work_joules / 4184.0) / 0.20
        
        # 3. Aerobic & Recovery Metabolic Expenditure (METs during rest intervals)
        rest_met = 2.5  # Elevated resting metabolic rate between sets
        rest_kcal = (rest_met * 3.5 * user_body_weight_kg / 200.0) * total_rest_time_min
        
        # Total Calories Burned for the exercise
        total_calories = metabolic_work_kcal + rest_kcal
        
        return {
            "exercise_name": exercise_name,
            "muscle_group": muscle_group,
            "sets": sets,
            "reps_per_set": reps,
            "total_reps": total_reps,
            "weight_lifted_kg": weight_lifted_kg,
            "total_volume_kg": round(sets * reps * weight_lifted_kg, 1),
            "mechanical_work_kcal": round(metabolic_work_kcal, 1),
            "rest_metabolic_kcal": round(rest_kcal, 1),
            "total_calories_burned": round(total_calories, 1),
            "estimated_duration_min": round(total_exercise_duration_min, 1)
        }

    def predict_routine_calories(
        self,
        exercises_list: list,
        user_body_weight_kg: float = 75.0
    ) -> dict:
        """
        Computes total calories burned for an entire workout session composed of multiple exercises.
        Each item in exercises_list is a dict:
        {"name": "Bench Press", "muscle_group": "chest", "sets": 4, "reps": 10, "weight_kg": 70, "rest_sec": 75}
        """
        exercise_breakdowns = []
        cumulative_calories = 0.0
        cumulative_duration_min = 0.0
        cumulative_volume_kg = 0.0
        
        for ex in exercises_list:
            res = self.predict_exercise_calories(
                exercise_name=ex.get("name", "Custom Exercise"),
                muscle_group=ex.get("muscle_group", "general"),
                sets=ex.get("sets", 3),
                reps=ex.get("reps", 10),
                weight_lifted_kg=ex.get("weight_kg", 20.0),
                user_body_weight_kg=user_body_weight_kg,
                rest_seconds_between_sets=ex.get("rest_sec", 60)
            )
            exercise_breakdowns.append(res)
            cumulative_calories += res["total_calories_burned"]
            cumulative_duration_min += res["estimated_duration_min"]
            cumulative_volume_kg += res["total_volume_kg"]
            
        return {
            "total_routine_calories": round(cumulative_calories, 1),
            "total_routine_duration_min": round(cumulative_duration_min, 1),
            "total_routine_volume_kg": round(cumulative_volume_kg, 1),
            "exercise_count": len(exercise_breakdowns),
            "exercises": exercise_breakdowns
        }

if __name__ == "__main__":
    predictor = CaloriePredictor()
    
    # Test session prediction
    session_res = predictor.predict_session_calories(
        age=26, gender="Male", weight_kg=78.0, height_m=1.78,
        session_duration_hours=1.2, workout_type="Strength", avg_bpm=148
    )
    print("Session Calorie Prediction Test:", session_res)
    
    # Test set-rep-weight exercise prediction
    ex_res = predictor.predict_exercise_calories(
        exercise_name="Barbell Squat",
        muscle_group="upper legs",
        sets=4,
        reps=8,
        weight_lifted_kg=100.0,
        user_body_weight_kg=78.0
    )
    print("Exercise Set/Rep/Weight Calorie Test:", ex_res)

