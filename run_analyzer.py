"""
Fitness and Workout Progress Analyzer
Script: run_analyzer.py
Description: Master application interface offering both automated demonstration
             and an interactive CLI for calorie prediction, workout planning,
             diet planning, and weight loss trajectory simulation.
"""

import os
import sys
import argparse
from tabulate import tabulate

# Ensure UTF-8 output encoding
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_gym_dataset
from src.dap_lab_experiments import run_all_experiments
from src.ml_models import train_and_evaluate_models
from src.calorie_predictor import CaloriePredictor
from src.planner_recommender import FitnessPlanner

def print_banner():
    banner = """
==============================================================================
   FITNESS AND WORKOUT PROGRESS ANALYZER (DAP LAB PROJECT)
   - Calorie Prediction Engine (ML: Linear Regression, Decision Tree, Gradient Boosting)
   - Exercise Specific Set/Rep/Weight Work Done Calculator
   - Automated Workout Plan Generator (Mapped to Exercise Database)
   - BMR / TDEE & Macronutrient Partitioned Diet Planner
   - 12-Week Weight Loss & Body Composition Simulator
==============================================================================
    """
    print(banner)

def demo_full_pipeline():
    """
    Executes a comprehensive, automated demonstration showcasing all capabilities.
    """
    print("\n[+] EXECUTING END-TO-END DEMONSTRATION FOR SAMPLE USER...")
    
    # Sample Athlete Profile
    user_profile = {
        "name": "Alex",
        "age": 27,
        "gender": "Male",
        "weight_kg": 82.5,
        "height_m": 1.78,
        "height_cm": 178.0,
        "workout_type": "Strength",
        "session_duration_hours": 1.25,
        "avg_bpm": 146,
        "max_bpm": 174,
        "resting_bpm": 62,
        "fat_percentage": 22.0,
        "water_intake_liters": 2.8,
        "workout_frequency_days": 4,
        "experience_level": 2,  # Intermediate
        "goal": "Weight Loss",
        "activity_level": "Moderately Active"
    }
    
    print("\n--- 1. USER PROFILE ---")
    u_table = [[k.replace("_", " ").title(), str(v)] for k, v in user_profile.items()]
    print(tabulate(u_table, headers=["Biometric / Fitness Metric", "Value"], tablefmt="grid"))
    
    # 1. Calorie Prediction via ML
    predictor = CaloriePredictor()
    pred_res = predictor.predict_session_calories(
        age=user_profile["age"],
        gender=user_profile["gender"],
        weight_kg=user_profile["weight_kg"],
        height_m=user_profile["height_m"],
        session_duration_hours=user_profile["session_duration_hours"],
        workout_type=user_profile["workout_type"],
        avg_bpm=user_profile["avg_bpm"],
        max_bpm=user_profile["max_bpm"],
        resting_bpm=user_profile["resting_bpm"],
        fat_percentage=user_profile["fat_percentage"],
        water_intake_liters=user_profile["water_intake_liters"],
        workout_frequency_days=user_profile["workout_frequency_days"],
        experience_level=user_profile["experience_level"]
    )
    
    print("\n--- 2. MACHINE LEARNING CALORIE BURN PREDICTION ---")
    c_table = [
        ["Model Architecture Employed", pred_res["model_architecture"]],
        ["Predicted Session Energy Burn", f"{pred_res['predicted_calories']} kcal"],
        ["Average Calorie Burn Rate", f"{pred_res['burn_rate_kcal_per_hour']} kcal / hour"],
        ["Computed Body Mass Index (BMI)", f"{pred_res['bmi']} kg/m²"]
    ]
    print(tabulate(c_table, headers=["Output Parameter", "Prediction"], tablefmt="grid"))
    
    # 2. Specific Resistance Training Exercise Burn
    print("\n--- 3. EXERCISE-SPECIFIC RESISTANCE SET/REP/WEIGHT CALCULATION ---")
    ex_routine = [
        {"name": "Barbell Squat", "muscle_group": "upper legs", "sets": 4, "reps": 10, "weight_kg": 90.0, "rest_sec": 75},
        {"name": "Barbell Bench Press", "muscle_group": "chest", "sets": 4, "reps": 10, "weight_kg": 75.0, "rest_sec": 60},
        {"name": "Bent-over Barbell Row", "muscle_group": "back", "sets": 4, "reps": 10, "weight_kg": 70.0, "rest_sec": 60},
        {"name": "Standing Overhead Press", "muscle_group": "shoulders", "sets": 3, "reps": 10, "weight_kg": 45.0, "rest_sec": 60}
    ]
    routine_res = predictor.predict_routine_calories(ex_routine, user_body_weight_kg=user_profile["weight_kg"])
    
    r_rows = []
    for ex in routine_res["exercises"]:
        r_rows.append([
            ex["exercise_name"],
            ex["muscle_group"].title(),
            f"{ex['sets']} x {ex['reps_per_set']}",
            f"{ex['weight_lifted_kg']} kg",
            f"{ex['total_volume_kg']} kg",
            f"{ex['mechanical_work_kcal']} kcal",
            f"{ex['total_calories_burned']} kcal"
        ])
    print(tabulate(r_rows, headers=["Exercise", "Target", "Sets x Reps", "Load", "Total Volume", "Work Burn", "Total Burn"], tablefmt="grid"))
    print(f">> Total Resistance Routine Caloric Burn: {routine_res['total_routine_calories']} kcal ({routine_res['total_routine_duration_min']} mins, Volume: {routine_res['total_routine_volume_kg']} kg)")
    
    # 3. Personalized Workout Plan
    planner = FitnessPlanner()
    workout_plan = planner.generate_workout_plan(
        goal=user_profile["goal"],
        experience_level="Intermediate",
        days_per_week=user_profile["workout_frequency_days"]
    )
    
    print("\n--- 4. PERSONALIZED WORKOUT PLAN (EXERCISE LIBRARY MAPPING) ---")
    print(f"Goal: {workout_plan['goal']} | Split Structure: {workout_plan['training_split']}")
    for day in workout_plan["schedule"]:
        print(f"\n* {day['day_name']} [Tempo: {day['tempo_guideline']}]:")
        d_rows = [[ex["exercise"], ex["target_muscle"], f"{ex['sets']} sets", ex["reps"], ex["rest"], ex["equipment"]] for ex in day["exercises"]]
        print(tabulate(d_rows, headers=["Exercise Name", "Target Muscle", "Sets", "Reps", "Rest Interval", "Equipment"], tablefmt="grid"))
        
    # 4. Custom Diet & Nutrition Plan
    diet_plan = planner.generate_diet_plan(
        age=user_profile["age"],
        gender=user_profile["gender"],
        weight_kg=user_profile["weight_kg"],
        height_cm=user_profile["height_cm"],
        activity_level=user_profile["activity_level"],
        fitness_goal=user_profile["goal"]
    )
    
    print("\n--- 5. PERSONALIZED DIET & MACRONUTRIENT NUTRITION PLAN ---")
    m = diet_plan["metabolic_rates"]
    macros = diet_plan["macronutrients"]
    print(f"Basal Metabolic Rate (BMR) : {m['bmr_kcal']} kcal/day")
    print(f"Total Daily Energy (TDEE)   : {m['tdee_kcal']} kcal/day")
    print(f"Target Caloric Intake       : {m['target_calories_kcal']} kcal/day ({m['strategy']})")
    print(f"Recommended Daily Hydration : {diet_plan['hydration_liters']} Liters/day\n")
    
    macro_table = [
        ["Protein", f"{macros['protein']['grams']} g", f"{macros['protein']['calories']} kcal", f"{macros['protein']['percent']}%"],
        ["Carbohydrates", f"{macros['carbohydrates']['grams']} g", f"{macros['carbohydrates']['calories']} kcal", f"{macros['carbohydrates']['percent']}%"],
        ["Fats", f"{macros['fats']['grams']} g", f"{macros['fats']['calories']} kcal", f"{macros['fats']['percent']}%"]
    ]
    print(tabulate(macro_table, headers=["Macronutrient", "Daily Grams", "Caloric Value", "Energy %"], tablefmt="grid"))
    
    print("\nMeal Distribution Blueprint:")
    meal_table = []
    for ml in diet_plan["meal_breakdown"]:
        meal_table.append([ml["meal"], f"{ml['calories']} kcal", f"{ml['protein_g']}g P / {ml['carbs_g']}g C / {ml['fat_g']}g F", ml["suggested_foods"]])
    print(tabulate(meal_table, headers=["Meal", "Calories", "Macros", "Nutrient-Dense Food Ideas"], tablefmt="grid"))
    
    # 5. Weight Loss Trajectory
    print("\n--- 6. 12-WEEK WEIGHT LOSS & BODY TRANSFORMATION PROJECTION ---")
    sim = planner.predict_weight_loss_trajectory(
        starting_weight_kg=user_profile["weight_kg"],
        height_cm=user_profile["height_cm"],
        age=user_profile["age"],
        gender=user_profile["gender"],
        daily_deficit_kcal=500.0,
        workout_burn_per_session=pred_res["predicted_calories"],
        workout_days_per_week=user_profile["workout_frequency_days"],
        total_weeks=12
    )
    print(f"Starting Weight         : {sim['starting_weight_kg']:.2f} kg")
    print(f"Projected Final Weight  : {sim['projected_final_weight_kg']:.2f} kg")
    print(f"Total Weight Lost       : -{sim['total_weight_lost_kg']:.2f} kg (Avg -{sim['weekly_average_loss_kg']:.2f} kg/week)")
    print(f"Final Projected BMI     : {sim['final_bmi']} kg/m²")
    print(f"Total Energy Expended   : {sim['total_energy_deficit_kcal']:,} kcal")
    print(f">> Trajectory chart saved to: {sim['plot_path']}\n")
    
    print(tabulate(sim["weekly_schedule_table"][:7], headers=["Week", "Projected Weight (kg)", "Net Change (kg)", "BMI", "Cum. Deficit (kcal)"], tablefmt="grid"))
    print("... (Weeks 7 to 12 simulated in full model output)\n")
    print("=" * 78)
    print(" END-TO-END DEMONSTRATION COMPLETED SUCCESSFULLY! ".center(78, "="))
    print("=" * 78)

def interactive_session_predict():
    """Interactive tool for session calorie prediction."""
    print("\n--- Interactive Session Calorie Prediction ---")
    try:
        age = int(input("Enter Age (e.g. 25): ") or 25)
        gender = input("Enter Gender (Male/Female) [Male]: ").strip().capitalize() or "Male"
        weight = float(input("Enter Weight in kg (e.g. 75.0): ") or 75.0)
        height = float(input("Enter Height in meters (e.g. 1.75): ") or 1.75)
        duration = float(input("Enter Session Duration in hours (e.g. 1.2): ") or 1.2)
        w_type = input("Enter Workout Type (Strength/Cardio/HIIT/Yoga) [Strength]: ").strip().capitalize() or "Strength"
        bpm = int(input("Enter Average Heart Rate in BPM (e.g. 145): ") or 145)
        
        predictor = CaloriePredictor()
        res = predictor.predict_session_calories(
            age=age, gender=gender, weight_kg=weight, height_m=height,
            session_duration_hours=duration, workout_type=w_type, avg_bpm=bpm
        )
        print("\n>>> PREDICTION RESULT:")
        print(f"    Estimated Calorie Burn : {res['predicted_calories']} kcal")
        print(f"    Model Employed         : {res['model_architecture']}")
        print(f"    Burn Rate              : {res['burn_rate_kcal_per_hour']} kcal/hour")
        print(f"    Body Mass Index (BMI)  : {res['bmi']} kg/m²")
    except Exception as e:
        print(f"Error during prediction: {e}")

def interactive_menu():
    """Interactive console menu for users."""
    print_banner()
    while True:
        print("\nMAIN MENU:")
        print("  1. Run Automated Full System Demonstration")
        print("  2. Predict Session Calorie Burn (Machine Learning)")
        print("  3. Calculate Exercise Resistance Work (Sets, Reps, Weights)")
        print("  4. Generate Custom Workout Plan (Exercise Library)")
        print("  5. Generate Custom Diet & Macronutrient Plan (BMR / TDEE)")
        print("  6. Simulate 12-Week Weight Loss Trajectory")
        print("  7. Execute all 8 DAP Lab Experiments (EXP 1 to EXP 8)")
        print("  8. Retrain Machine Learning Models")
        print("  9. Exit")
        
        choice = input("\nEnter choice (1-9) [1]: ").strip() or "1"
        
        if choice == "1":
            demo_full_pipeline()
        elif choice == "2":
            interactive_session_predict()
        elif choice == "3":
            predictor = CaloriePredictor()
            ex_name = input("Exercise Name [Bench Press]: ") or "Bench Press"
            muscle = input("Muscle Group (chest/upper legs/back/shoulders/upper arms/waist) [chest]: ") or "chest"
            sets = int(input("Sets [4]: ") or 4)
            reps = int(input("Reps [10]: ") or 10)
            weight = float(input("Weight in kg [70]: ") or 70.0)
            bw = float(input("Body Weight in kg [75]: ") or 75.0)
            res = predictor.predict_exercise_calories(ex_name, muscle, sets, reps, weight, bw)
            print(f"\n>>> Energy Expended for {ex_name}: {res['total_calories_burned']} kcal (Volume: {res['total_volume_kg']} kg)")
        elif choice == "4":
            planner = FitnessPlanner()
            goal = input("Goal (Weight Loss/Muscle Building/Strength) [Weight Loss]: ") or "Weight Loss"
            days = int(input("Days per week (3-6) [4]: ") or 4)
            plan = planner.generate_workout_plan(goal=goal, days_per_week=days)
            print(f"\nGenerated {plan['training_split']}:")
            for day in plan["schedule"]:
                print(f"\n* {day['day_name']}:")
                for ex in day["exercises"]:
                    print(f"  - {ex['exercise']} ({ex['target_muscle']}): {ex['sets']} sets x {ex['reps']}")
        elif choice == "5":
            planner = FitnessPlanner()
            w = float(input("Weight in kg [80]: ") or 80.0)
            h = float(input("Height in cm [175]: ") or 175.0)
            a = int(input("Age [26]: ") or 26)
            g = input("Gender (Male/Female) [Male]: ") or "Male"
            goal = input("Goal (Weight Loss/Muscle Building/Maintenance) [Weight Loss]: ") or "Weight Loss"
            diet = planner.generate_diet_plan(age=a, gender=g, weight_kg=w, height_cm=h, fitness_goal=goal)
            print(f"\nTarget Daily Calories: {diet['metabolic_rates']['target_calories_kcal']} kcal")
            print(f"Macros: {diet['macronutrients']['protein']['grams']}g Protein | {diet['macronutrients']['carbohydrates']['grams']}g Carbs | {diet['macronutrients']['fats']['grams']}g Fats")
        elif choice == "6":
            planner = FitnessPlanner()
            w = float(input("Starting Weight in kg [85]: ") or 85.0)
            sim = planner.predict_weight_loss_trajectory(starting_weight_kg=w)
            print(f"\nProjected Final Weight after 12 Weeks: {sim['projected_final_weight_kg']} kg (-{sim['total_weight_lost_kg']} kg)")
            print(f"Plot saved to: {sim['plot_path']}")
        elif choice == "7":
            df = load_gym_dataset()
            run_all_experiments(df)
        elif choice == "8":
            df = load_gym_dataset()
            train_and_evaluate_models(df)
        elif choice == "9":
            print("\nExiting Fitness and Workout Progress Analyzer. Goodbye!")
            break
        else:
            print("Invalid selection. Please choose an option from 1 to 9.")

def main():
    parser = argparse.ArgumentParser(description="Fitness and Workout Progress Analyzer")
    parser.add_argument("--demo", action="store_true", help="Run full automated system demonstration")
    parser.add_argument("--experiments", action="store_true", help="Execute all 8 DAP Lab experiments")
    parser.add_argument("--train", action="store_true", help="Train and benchmark ML models")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal CLI menu")
    
    args = parser.parse_args()
    
    if args.experiments:
        df = load_gym_dataset()
        run_all_experiments(df)
    elif args.train:
        df = load_gym_dataset()
        train_and_evaluate_models(df)
    elif args.demo:
        print_banner()
        demo_full_pipeline()
    elif args.interactive:
        interactive_menu()
    else:
        # Default behavior: run demonstration
        print_banner()
        demo_full_pipeline()

if __name__ == "__main__":
    main()

