"""
Fitness and Workout Progress Analyzer
Module: planner_recommender.py
Description: Generates personalized workout plans (using exercises.json),
             scientifically balanced diet plans (BMR/TDEE & Macros),
             and dynamic 12-week weight loss trajectories.
"""

import os
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import base64
import numpy as np
import pandas as pd
import matplotlib
if "ipykernel" not in sys.modules and "IPython" not in sys.modules:
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

try:
    from src.data_loader import load_exercises_dataset
except ModuleNotFoundError:
    from data_loader import load_exercises_dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "output_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

class FitnessPlanner:
    def __init__(self, exercises_data: list = None):
        self.exercises = exercises_data if exercises_data is not None else load_exercises_dataset()
        self._index_exercises()

    def _index_exercises(self):
        """Index exercises by body_part and target for fast lookup."""
        self.by_body_part = {}
        for ex in self.exercises:
            bp = ex.get("body_part", "other")
            if bp not in self.by_body_part:
                self.by_body_part[bp] = []
            self.by_body_part[bp].append(ex)

    def _filter_exercise(self, body_part: str, equipment_pref: str = "all", preferred_target: str = None):
        """Finds exercises matching body part and equipment."""
        candidates = self.by_body_part.get(body_part, [])
        if not candidates:
            return None
        
        filtered = candidates
        if equipment_pref != "all":
            filtered = [e for e in filtered if equipment_pref.lower() in (e.get("equipment") or "").lower()]
            if not filtered:
                filtered = candidates
                
        if preferred_target:
            target_match = [e for e in filtered if preferred_target.lower() in (e.get("target") or "").lower()]
            if target_match:
                filtered = target_match
                
        # Pick representative clean exercise
        chosen = filtered[np.random.randint(0, len(filtered))]
        instr = chosen.get("instructions", {})
        if isinstance(instr, dict):
            cue = instr.get("en", "Maintain steady cadence and controlled form.")
        elif isinstance(instr, list):
            cue = " ".join(instr[:2])
        elif isinstance(instr, str):
            cue = instr
        else:
            cue = "Maintain steady cadence and controlled form."

        img_rel = chosen.get("image", "")
        gif_rel = chosen.get("gif_url", "")
        img_path = f"datasets/exercises-dataset-main/{img_rel}".replace("\\", "/") if img_rel else ""
        gif_path = f"datasets/exercises-dataset-main/{gif_rel}".replace("\\", "/") if gif_rel else ""

        return {
            "name": chosen.get("name", "Standard Exercise").title(),
            "target": chosen.get("target", "general"),
            "body_part": body_part,
            "equipment": chosen.get("equipment", "general"),
            "instructions": cue,
            "image_path": img_path,
            "gif_path": gif_path
        }

    # =========================================================================
    # 1. WORKOUT PLAN GENERATOR
    # =========================================================================
    def generate_workout_plan(
        self,
        goal: str = "Weight Loss",
        experience_level: str = "Intermediate",
        days_per_week: int = 4,
        equipment: str = "all"
    ) -> dict:
        """
        Creates a custom multi-day workout split tailored to fitness goals,
        experience level, and days per week, utilizing exercises from the dataset.
        """
        np.random.seed(42)  # For reproducible sample generation
        
        # Set rep & rest configurations based on Goal
        if goal.lower() in ["weight loss", "fat loss"]:
            rep_range = "12-15 reps"
            sets_count = 3 if experience_level == "Beginner" else 4
            rest_interval = "45-60 sec"
            tempo = "Moderate-High Pace (Metabolic Conditioning)"
        elif goal.lower() in ["muscle building", "hypertrophy"]:
            rep_range = "8-12 reps"
            sets_count = 3 if experience_level == "Beginner" else 4
            rest_interval = "75-90 sec"
            tempo = "Controlled (3s eccentric, 1s concentric)"
        else:  # Strength
            rep_range = "4-6 reps"
            sets_count = 4 if experience_level == "Beginner" else 5
            rest_interval = "120-180 sec"
            tempo = "Explosive concentric"

        plan = {
            "goal": goal,
            "experience_level": experience_level,
            "days_per_week": days_per_week,
            "equipment": equipment,
            "training_split": "",
            "schedule": []
        }

        # Determine Split Architecture
        if days_per_week <= 3:
            plan["training_split"] = "3-Day Full Body Circuit"
            days_templates = [
                ("Day 1: Full Body A (Chest, Upper Legs, Back, Core)", ["chest", "upper legs", "back", "waist"]),
                ("Day 2: Full Body B (Shoulders, Hamstrings, Upper Arms, Cardio)", ["shoulders", "upper legs", "upper arms", "cardio"]),
                ("Day 3: Full Body C (Compound Power & Core Stability)", ["back", "chest", "upper legs", "waist"])
            ]
        elif days_per_week == 4:
            plan["training_split"] = "4-Day Upper / Lower Split"
            days_templates = [
                ("Day 1: Upper Body Strength (Chest, Back, Shoulders)", ["chest", "back", "shoulders", "upper arms"]),
                ("Day 2: Lower Body & Core (Quads, Hamstrings, Calves, Abs)", ["upper legs", "upper legs", "lower legs", "waist"]),
                ("Day 3: Upper Body Hypertrophy (Back, Chest, Arms)", ["back", "chest", "upper arms", "waist"]),
                ("Day 4: Lower Body & Conditioning (Legs, Glutes, Cardio)", ["upper legs", "upper legs", "cardio", "waist"])
            ]
        else:  # 5 or 6 days
            plan["training_split"] = "5/6-Day Push / Pull / Legs (PPL) Split"
            days_templates = [
                ("Day 1: Push A (Chest, Front Deltoids, Triceps)", ["chest", "shoulders", "upper arms", "chest"]),
                ("Day 2: Pull A (Lats, Traps, Rear Deltoids, Biceps)", ["back", "back", "upper arms", "waist"]),
                ("Day 3: Legs A (Quads, Glutes, Hamstrings, Calves)", ["upper legs", "upper legs", "lower legs", "waist"]),
                ("Day 4: Push B (Overhead Deltoids, Incline Chest, Core)", ["shoulders", "chest", "upper arms", "waist"]),
                ("Day 5: Pull B (Deadlift Pattern, Mid-Back, Biceps)", ["back", "back", "upper arms", "cardio"]),
                ("Day 6: Legs B (Posterior Chain, Glute Isolation, Cardio)", ["upper legs", "upper legs", "cardio", "waist"])
            ][:days_per_week]

        for day_title, target_body_parts in days_templates:
            day_exercises = []
            for bp in target_body_parts:
                ex_info = self._filter_exercise(bp, equipment)
                if ex_info:
                    day_exercises.append({
                        "exercise": ex_info["name"],
                        "target_muscle": ex_info["target"].title(),
                        "body_part": ex_info["body_part"].title(),
                        "equipment": ex_info["equipment"].title(),
                        "sets": sets_count,
                        "reps": rep_range,
                        "rest": rest_interval,
                        "technique_cue": ex_info["instructions"][:110] + "...",
                        "image_path": ex_info.get("image_path", ""),
                        "gif_path": ex_info.get("gif_path", "")
                    })
            plan["schedule"].append({
                "day_name": day_title,
                "tempo_guideline": tempo,
                "exercises": day_exercises
            })

        return plan

    def find_exercise_by_name(self, name_query: str) -> dict:
        """
        Searches exercises dataset by name and returns matching exercise with media paths.
        """
        q = name_query.lower().strip()
        for ex in self.exercises:
            if q in (ex.get("name") or "").lower():
                img_rel = ex.get("image", "")
                gif_rel = ex.get("gif_url", "")
                instr = ex.get("instructions", {})
                cue = instr.get("en", "Maintain steady cadence.") if isinstance(instr, dict) else str(instr)
                return {
                    "name": ex.get("name", "Exercise").title(),
                    "target": (ex.get("target") or "general").title(),
                    "body_part": (ex.get("body_part") or "general").title(),
                    "equipment": (ex.get("equipment") or "general").title(),
                    "instructions": cue,
                    "image_path": f"datasets/exercises-dataset-main/{img_rel}".replace("\\", "/") if img_rel else "",
                    "gif_path": f"datasets/exercises-dataset-main/{gif_rel}".replace("\\", "/") if gif_rel else ""
                }
        return None

    @staticmethod
    def get_media_data_uri(filepath: str) -> str:
        """
        Converts a local image or GIF file into a base64 Data URI.
        Ensures guaranteed rendering across Jupyter Notebooks, VS Code webviews,
        and offline HTML exports without relative path resolution or CORS errors.
        """
        if not filepath:
            return ""
        if str(filepath).startswith("data:") or str(filepath).startswith("http"):
            return filepath

        clean_path = str(filepath).replace("\\", "/")
        candidates = [
            clean_path,
            os.path.join(PROJECT_ROOT, clean_path),
            os.path.join(PROJECT_ROOT, "datasets", clean_path),
        ]
        if "exercises-dataset-main" in clean_path:
            rel = clean_path.split("exercises-dataset-main/", 1)[-1]
            candidates.append(os.path.join(PROJECT_ROOT, "datasets", "exercises-dataset-main", rel))
            candidates.append(os.path.join(PROJECT_ROOT, "exercises-dataset-main", rel))
        else:
            candidates.append(os.path.join(PROJECT_ROOT, "datasets", "exercises-dataset-main", clean_path))

        resolved = None
        for c in candidates:
            if c and os.path.exists(c) and os.path.isfile(c):
                resolved = c
                break

        if not resolved:
            return filepath

        ext = os.path.splitext(resolved)[1].lower()
        mime = "image/gif" if ext == ".gif" else ("image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png")
        try:
            with open(resolved, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
        except Exception:
            return filepath

    def render_workout_cards_html(self, plan: dict, max_days: int = 2) -> str:
        """
        Generates an HTML representation of the workout plan with animated GIF cards
        and static thumbnail images designed for rich display in Jupyter Notebooks.
        """
        html_blocks = []
        html_blocks.append('<div style="font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin-bottom: 25px;">')
        html_blocks.append(f'<h3 style="color: #1e3a8a; margin-bottom: 6px;">🏋️ Recommended Workout Split: {plan["training_split"]}</h3>')
        html_blocks.append(f'<p style="color: #64748b; margin-top: 0; font-size: 14px;">Goal: <b>{plan["goal"]}</b> | Level: <b>{plan["experience_level"]}</b> | Days/Week: <b>{plan["days_per_week"]}</b></p>')

        for day in plan["schedule"][:max_days]:
            html_blocks.append(f'<div style="background: #f1f5f9; border-left: 4px solid #2563eb; padding: 10px 14px; margin: 16px 0 10px 0; border-radius: 4px;">')
            html_blocks.append(f'<h4 style="margin: 0; color: #1e40af; font-size: 15px;">{day["day_name"]}</h4>')
            html_blocks.append(f'<small style="color: #64748b;">Tempo: {day["tempo_guideline"]}</small>')
            html_blocks.append('</div>')

            html_blocks.append('<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px; margin-bottom: 18px;">')
            for ex in day["exercises"]:
                gif_src = ex.get("gif_path", "")
                img_src = ex.get("image_path", "")
                gif_uri = self.get_media_data_uri(gif_src)
                img_uri = self.get_media_data_uri(img_src)

                media_tag = f"""
                <div style="display: flex; gap: 8px; flex-shrink: 0; align-items: center;">
                    <div style="text-align: center;">
                        <img src="{img_uri}" alt="{ex['exercise']} Form" title="Start/End Posture (images/)" style="width: 75px; height: 75px; object-fit: cover; border-radius: 6px; border: 1px solid #cbd5e1;" />
                        <div style="font-size: 9px; color: #64748b; font-weight: 600; margin-top: 2px;">Form 📷</div>
                    </div>
                    <div style="text-align: center;">
                        <img src="{gif_uri}" alt="{ex['exercise']} Motion" title="Execution Loop (videos/)" style="width: 75px; height: 75px; object-fit: cover; border-radius: 6px; border: 1.5px solid #2563eb;" />
                        <div style="font-size: 9px; color: #2563eb; font-weight: 600; margin-top: 2px;">Motion 🎬</div>
                    </div>
                </div>
                """

                card_html = f"""
                <div style="display: flex; align-items: center; background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    {media_tag}
                    <div style="margin-left: 14px; flex-grow: 1;">
                        <div style="font-weight: 700; font-size: 15px; color: #0f172a; margin-bottom: 4px;">{ex["exercise"]}</div>
                        <div style="display: flex; gap: 6px; margin-bottom: 6px;">
                            <span style="background: #dbeafe; color: #1d4ed8; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 12px;">{ex["target_muscle"]}</span>
                            <span style="background: #f1f5f9; color: #475569; font-size: 11px; padding: 2px 8px; border-radius: 12px;">{ex["equipment"]}</span>
                        </div>
                        <div style="font-size: 13px; color: #1e293b; margin-bottom: 3px;">
                            <b>Prescription:</b> {ex["sets"]} sets &times; {ex["reps"]}
                        </div>
                        <div style="font-size: 12px; color: #64748b;">
                            <b>Rest:</b> {ex["rest"]} | <small>{ex["technique_cue"][:55]}...</small>
                        </div>
                    </div>
                </div>
                """
                html_blocks.append(card_html)
            html_blocks.append('</div>')

        html_blocks.append('</div>')
        return "".join(html_blocks)

    # =========================================================================
    # 2. DIET PLAN GENERATOR
    # =========================================================================
    def generate_diet_plan(
        self,
        age: int = 28,
        gender: str = "Male",
        weight_kg: float = 78.0,
        height_cm: float = 175.0,
        activity_level: str = "Moderately Active",
        fitness_goal: str = "Weight Loss"
    ) -> dict:
        """
        Computes Mifflin-St Jeor Basal Metabolic Rate (BMR), Total Daily Energy
        Expenditure (TDEE), and custom macronutrient profile (Protein, Carbs, Fats).
        """
        # 1. BMR Calculation (Mifflin-St Jeor Equation)
        if gender.lower() == "male":
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
        else:
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
            
        # 2. Physical Activity Level (PAL) Multiplier
        activity_multipliers = {
            "sedentary": 1.2,
            "lightly active": 1.375,
            "moderately active": 1.55,
            "very active": 1.725,
            "extra active": 1.9
        }
        pal = activity_multipliers.get(activity_level.lower(), 1.55)
        tdee = bmr * pal
        
        # 3. Caloric Target based on Fitness Goal
        goal_lower = fitness_goal.lower()
        if "weight loss" in goal_lower or "fat loss" in goal_lower:
            deficit_kcal = 500  # Safe sustained caloric deficit
            target_calories = tdee - deficit_kcal
            goal_desc = "Caloric Deficit (-500 kcal/day for ~0.45 kg weekly fat loss)"
        elif "muscle building" in goal_lower or "gain" in goal_lower:
            surplus_kcal = 300  # Clean lean surplus
            target_calories = tdee + surplus_kcal
            goal_desc = "Lean Caloric Surplus (+300 kcal/day for lean hypertrophy)"
        else:
            target_calories = tdee
            goal_desc = "Caloric Maintenance (Energy equilibrium)"
            
        # 4. Macronutrient Partitioning
        # Protein: 2.0g per kg bodyweight (essential for muscle preservation during deficit)
        protein_g = round(weight_kg * 2.0, 1)
        protein_kcal = protein_g * 4.0
        
        # Fats: 25% of target caloric intake
        fat_kcal = target_calories * 0.25
        fat_g = round(fat_kcal / 9.0, 1)
        
        # Carbohydrates: Remaining calories
        carb_kcal = max(target_calories - (protein_kcal + fat_kcal), 0.0)
        carb_g = round(carb_kcal / 4.0, 1)
        
        # Percentages
        p_pct = round((protein_kcal / target_calories) * 100, 1)
        f_pct = round((fat_kcal / target_calories) * 100, 1)
        c_pct = round((carb_kcal / target_calories) * 100, 1)
        
        # Daily Meal Distribution
        meals = [
            {
                "meal": "Breakfast (25%)",
                "calories": round(target_calories * 0.25),
                "protein_g": round(protein_g * 0.25),
                "carbs_g": round(carb_g * 0.25),
                "fat_g": round(fat_g * 0.25),
                "suggested_foods": "Oatmeal with whey protein, chia seeds, and sliced berries OR Scrambled egg whites with whole grain toast."
            },
            {
                "meal": "Lunch (35%)",
                "calories": round(target_calories * 0.35),
                "protein_g": round(protein_g * 0.35),
                "carbs_g": round(carb_g * 0.35),
                "fat_g": round(fat_g * 0.35),
                "suggested_foods": "Grilled chicken breast / Tofu bowl with brown rice, steamed broccoli, and avocado slice."
            },
            {
                "meal": "Pre/Post Workout Snack (15%)",
                "calories": round(target_calories * 0.15),
                "protein_g": round(protein_g * 0.15),
                "carbs_g": round(carb_g * 0.15),
                "fat_g": round(fat_g * 0.15),
                "suggested_foods": "Greek yogurt with honey and almonds OR Protein shake with a ripe banana."
            },
            {
                "meal": "Dinner (25%)",
                "calories": round(target_calories * 0.25),
                "protein_g": round(protein_g * 0.25),
                "carbs_g": round(carb_g * 0.25),
                "fat_g": round(fat_g * 0.25),
                "suggested_foods": "Baked salmon / Lentil curry with quinoa, roasted asparagus, and light olive oil dressing."
            }
        ]
        
        water_intake_recommendation = round(weight_kg * 0.035, 2)  # ~35 ml/kg/day
        
        return {
            "user_biometrics": {
                "age": age,
                "gender": gender,
                "weight_kg": weight_kg,
                "height_cm": height_cm,
                "activity_level": activity_level
            },
            "metabolic_rates": {
                "bmr_kcal": round(bmr, 1),
                "tdee_kcal": round(tdee, 1),
                "target_calories_kcal": round(target_calories, 1),
                "strategy": goal_desc
            },
            "macronutrients": {
                "protein": {"grams": protein_g, "calories": round(protein_kcal), "percent": p_pct},
                "carbohydrates": {"grams": carb_g, "calories": round(carb_kcal), "percent": c_pct},
                "fats": {"grams": fat_g, "calories": round(fat_kcal), "percent": f_pct}
            },
            "hydration_liters": water_intake_recommendation,
            "meal_breakdown": meals
        }

    # =========================================================================
    # 3. WEIGHT LOSS & PROGRESS TRAJECTORY SIMULATOR
    # =========================================================================
    def predict_weight_loss_trajectory(
        self,
        starting_weight_kg: float = 85.0,
        height_cm: float = 175.0,
        age: int = 28,
        gender: str = "Male",
        daily_deficit_kcal: float = 500.0,
        workout_burn_per_session: float = 450.0,
        workout_days_per_week: int = 4,
        total_weeks: int = 12,
        plots_dir: str = PLOTS_DIR
    ) -> dict:
        """
        Simulates 12-week weight loss and fat mass reduction trajectory.
        Includes metabolic adaptation (gradual downward adjustment of BMR as body mass decreases).
        """
        # Physics constant: 1 kg body fat ~ 7700 kcal
        KCAL_PER_KG_FAT = 7700.0
        
        weeks = list(range(total_weeks + 1))
        weight_history = [starting_weight_kg]
        bmi_history = [round(starting_weight_kg / ((height_cm / 100) ** 2), 2)]
        cum_calories_burned = [0.0]
        
        current_weight = starting_weight_kg
        total_expended = 0.0
        
        weekly_table = []
        weekly_table.append([0, f"{starting_weight_kg:.2f}", "0.00", f"{bmi_history[0]:.2f}", "0"])
        
        for w in range(1, total_weeks + 1):
            # Dynamic metabolic adaptation: BMR decreases by ~15 kcal/day per kg lost
            weight_lost_so_far = starting_weight_kg - current_weight
            adapted_daily_deficit = max(daily_deficit_kcal - (weight_lost_so_far * 15.0), 200.0)
            
            # Weekly Total Deficit = (Daily Diet Deficit * 7) + (Workout Burn * Days)
            weekly_diet_deficit = adapted_daily_deficit * 7
            weekly_workout_burn = workout_burn_per_session * workout_days_per_week
            weekly_total_deficit = weekly_diet_deficit + weekly_workout_burn
            
            total_expended += weekly_total_deficit
            weight_loss_kg = weekly_total_deficit / KCAL_PER_KG_FAT
            
            current_weight = max(current_weight - weight_loss_kg, 40.0)
            current_bmi = round(current_weight / ((height_cm / 100) ** 2), 2)
            
            weight_history.append(round(current_weight, 2))
            bmi_history.append(current_bmi)
            cum_calories_burned.append(round(total_expended))
            
            cum_loss = starting_weight_kg - current_weight
            weekly_table.append([w, f"{current_weight:.2f}", f"-{cum_loss:.2f}", f"{current_bmi:.2f}", f"{int(total_expended):,}"])
            
        total_lost = starting_weight_kg - current_weight
        
        # Visualization: Weight Loss Trajectory & Cumulative Energy Deficit
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("12-Week Weight Loss & Fitness Progress Projection", fontsize=15, weight="bold")
        
        # Subplot 1: Body Weight & BMI Progression
        axes[0].plot(weeks, weight_history, marker="o", color="#2980b9", linewidth=2.5, label="Projected Weight (kg)")
        axes[0].set_title(f"Projected Body Weight Curve (Starting: {starting_weight_kg}kg -> Goal: {current_weight:.1f}kg)")
        axes[0].set_xlabel("Weeks of Training & Nutrition")
        axes[0].set_ylabel("Body Weight (kg)", color="#2980b9", weight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        
        ax0_twin = axes[0].twinx()
        ax0_twin.plot(weeks, bmi_history, marker="s", color="#e67e22", linestyle=":", linewidth=2, label="BMI")
        ax0_twin.set_ylabel("BMI (kg/m²)", color="#e67e22", weight="bold")
        ax0_twin.grid(False)
        
        # Subplot 2: Cumulative Energy Deficit (kcal)
        axes[1].bar(weeks[1:], [cum_calories_burned[i] for i in range(1, len(weeks))], color="#27ae60", alpha=0.75, width=0.6)
        axes[1].set_title(f"Cumulative Caloric Deficit (Total: {int(total_expended):,} kcal)")
        axes[1].set_xlabel("Weeks of Training")
        axes[1].set_ylabel("Cumulative Energy Deficit (kcal)")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "weight_loss_trajectory.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        return {
            "starting_weight_kg": starting_weight_kg,
            "projected_final_weight_kg": round(current_weight, 2),
            "total_weight_lost_kg": round(total_lost, 2),
            "weekly_average_loss_kg": round(total_lost / total_weeks, 2),
            "final_bmi": bmi_history[-1],
            "total_energy_deficit_kcal": int(total_expended),
            "weekly_schedule_table": weekly_table,
            "plot_path": plot_path
        }

if __name__ == "__main__":
    planner = FitnessPlanner()
    
    # Test Workout Plan
    plan = planner.generate_workout_plan(goal="Weight Loss", days_per_week=4)
    print("=== WORKOUT PLAN PREVIEW ===")
    print("Split:", plan["training_split"])
    for day in plan["schedule"][:2]:
        print(f"\n{day['day_name']} (Tempo: {day['tempo_guideline']}):")
        for ex in day["exercises"]:
            print(f"  - {ex['exercise']} ({ex['target_muscle']}) | {ex['sets']} sets x {ex['reps']} | Rest: {ex['rest']}")
            
    # Test Diet Plan
    diet = planner.generate_diet_plan(weight_kg=80, height_cm=176, age=27, fitness_goal="Weight Loss")
    print("\n=== DIET PLAN PREVIEW ===")
    print(f"BMR: {diet['metabolic_rates']['bmr_kcal']} kcal | TDEE: {diet['metabolic_rates']['tdee_kcal']} kcal | Target: {diet['metabolic_rates']['target_calories_kcal']} kcal")
    print(f"Macros: Protein {diet['macronutrients']['protein']['grams']}g | Carbs {diet['macronutrients']['carbohydrates']['grams']}g | Fats {diet['macronutrients']['fats']['grams']}g")
    
    # Test Weight Loss Simulation
    sim = planner.predict_weight_loss_trajectory(starting_weight_kg=84.0, height_cm=175.0)
    print("\n=== WEIGHT LOSS PROJECTION ===")
    print(f"Initial: {sim['starting_weight_kg']} kg -> Final: {sim['projected_final_weight_kg']} kg (-{sim['total_weight_lost_kg']} kg)")
