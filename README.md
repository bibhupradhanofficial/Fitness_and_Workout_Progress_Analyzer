# Fitness and Workout Progress Analyzer
> **Data Analysis with Python (DAP) Lab — Complete End-to-End Project & Machine Learning Suite**

---

## 📌 Executive Summary & Objectives

The **Fitness and Workout Progress Analyzer** is an end-to-end Python data science and machine learning system engineered to analyze fitness biometrics, predict workout energy expenditure, and generate evidence-based exercise, nutrition, and weight-loss trajectories.

The project fulfills two primary functional goals while strictly implementing the entire 8-experiment **Data Analysis with Python (DAP) Lab syllabus**:

1. **Goal 1: Calorie Burn Prediction Engine**
   - **Session-Level Machine Learning:** Predict total session caloric burn using athlete biometrics, workout modality, duration, and cardiovascular telemetry.
   - **Resistance-Exercise Set/Rep/Weight Mechanical Engine:** Calculates work done ($W = m \cdot g \cdot h \cdot \text{reps}$) and metabolic recovery energy across muscle groups.
2. **Goal 2: Workout Plan, Diet Plan & Weight Loss Predictor**
   - **Intelligent Workout Split Generator:** Dynamically compiles multi-day routines using an integrated library of **1,324 exercises** mapped by target muscle group, body part, and equipment.
   - **Precision Diet & Macro Recommender:** Computes Basal Metabolic Rate (BMR via Mifflin-St Jeor), Total Daily Energy Expenditure (TDEE), target caloric deficit/surplus, and gram-level macronutrient partitions.
   - **Dynamic 12-Week Weight Loss Simulator:** Models week-by-week weight, BMI, and body fat reduction with metabolic adaptation feedback.

---

## 🔬 Syllabus Mapping: DAP Lab Experiments (EXP 1 to EXP 8)

| Experiment | Academic Syllabus Objective | Project Implementation & Methodology | Output Artifact |
|---|---|---|---|
| **EXP 1: Data Handling** | Load DataFrame, classify/tabulate dtypes, histograms, scatter plots, box plots | Ingests `gym_members_exercise_tracking.csv` (973 rows, 15 cols), classifies continuous/discrete/categorical variables, visualizes distributions. | `output_plots/exp1_data_handling.png` |
| **EXP 2: Central Tendency & Variability** | Compute Mean, Median, Mode, Quartiles, IQR, Variance, Std Dev, Box Plot | Computes parametric/non-parametric dispersion across features; identifies outliers using the $1.5 \times \text{IQR}$ rule; produces annotated box plots. | `output_plots/exp2_variability.png` |
| **EXP 3: Transformations & Weighted Averages** | Log transformation, Geometric Mean, Weighted Mean, analyze distribution shifts | Applies natural log $\ln(x)$ on skewed calories, proves $\text{AM} \ge \text{GM}$, weights calories by session duration & frequency; evaluates skewness/kurtosis changes. | `output_plots/exp3_transformations.png` |
| **EXP 4: Correlation Analysis** | Compute Pearson $r$ and Spearman $\rho$, scatter plots, heatmaps, statistical interpretation | Generates lower-triangle correlation matrix; compares linear ($r$) vs. monotonic ($\rho$) rank relationships between duration, BPM, weight, and calories. | `output_plots/exp4_correlation.png` |
| **EXP 5: Regression** | Build simple linear regression model, compare correlation vs. regression | Fits OLS model ($\hat{y} = \beta_0 + \beta_1 x$); proves $R^2 = r^2$ and $\beta_1 = r \frac{s_y}{s_x}$; plots regression trendline with residual diagnostics. | `output_plots/exp5_linear_regression.png` |
| **EXP 6: Sampling & Hypothesis Testing** | Simple Random Sampling, Central Limit Theorem, Two-sample t-test | Simulates SRS with $n \in [30, 75, 150, 300]$; runs Welch's two-sample t-test testing gender & modality calorie differences ($p < 0.05$). | `output_plots/exp6_hypothesis_testing.png` |
| **EXP 7: NumPy Fundamentals** | NumPy aggregations, multi-axis slicing, boolean masking, sorting | Converts data to ndarrays; demonstrates `np.mean`, `np.std`, `np.percentile`, boolean masking for high-intensity sessions, and vectorized MET burn rate. | Console Output / Notebook |
| **EXP 8: Advanced Manipulation & Visualization** | Missing data management, hierarchical MultiIndex, 3D Matplotlib surface | Simulates MCAR missingness (dropna vs median imputation); creates MultiIndex grouping on `[Workout_Type, Gender]`; renders 3D regression surface plane. | `output_plots/exp8_3d_regression_surface.png`, `output_plots/exp8_hierarchical_analysis.png` |

---

## 🤖 Machine Learning Pipeline & Comparative Benchmark

We evaluated multiple regression architectures to predict `Calories_Burned` using an 80/20 train/test split with **5-Fold Cross Validation** and standard feature scaling (`StandardScaler` + `OneHotEncoder`).

### Model Performance Benchmark

| Model Architecture | Test MAE (kcal) | Test RMSE (kcal) | Test $R^2$ Score | 5-Fold CV $R^2$ (Mean $\pm$ Std) | Verdict |
|---|---|---|---|---|---|
| **Gradient Boosting Regressor** | **16.73** | **23.88** | **0.9932** | **0.9920 $\pm$ 0.0020** | 🏆 **Best Model (Selected)** |
| **Random Forest Regressor** | 37.59 | 49.20 | 0.9710 | 0.9666 $\pm$ 0.0049 | Strong Ensemble |
| **Multiple Linear Regression** | 30.27 | 40.57 | 0.9803 | 0.9774 $\pm$ 0.0033 | Highly Interpretable Linear Baseline |
| **Decision Tree Regressor** | 57.89 | 74.30 | 0.9338 | 0.9124 $\pm$ 0.0170 | Pruned Tree Benchmark |

### Feature Importance (Gradient Boosting)
1. **Session Duration (hours)**: **82.56%** (Primary driver of total thermodynamic expenditure)
2. **Average Heart Rate (Avg BPM)**: **10.90%** (Intensity and cardiovascular strain)
3. **Age**: **2.64%** (Metabolic efficiency decline factor)
4. **Gender**: **1.84%** (Lean muscle mass distribution)
5. **Body Fat Percentage**: **1.65%** (Active metabolic tissue ratio)

*All models are serialized in `models/best_model_bundle.joblib`.*

---

## 🏋️ Goal 1: Calorie Burn Prediction Engine

### 1. Full Workout Session Prediction
Inputs athlete age, gender, weight, height, workout type (Strength, Cardio, HIIT, Yoga), duration, and heart rate telemetry into the trained Gradient Boosting pipeline.

### 2. Resistance Training Mechanical Work & Recovery Equation
Computes mechanical work for strength exercises:
$$W_{\text{mech}} = \sum (\text{sets} \times \text{reps} \times \text{load (kg)} \times 9.81 \, \text{m/s}^2 \times \text{ROM (m)})$$
Accounting for human muscular thermodynamic efficiency ($\eta \approx 20\%$):
$$E_{\text{work (kcal)}} = \frac{W_{\text{mech}}}{0.20 \times 4184}$$
Plus elevated recovery metabolic expenditure during rest intervals:
$$E_{\text{rest (kcal)}} = \left(\text{MET}_{\text{rest}} \times 3.5 \times \frac{\text{Weight}_{\text{kg}}}{200}\right) \times \text{Rest Time}_{\text{min}}$$

---

## 🥗 Goal 2: Workout Plan, Diet Plan & Weight Loss

### 1. Workout Plan Generator
- Powered by `datasets/exercises-dataset-main/data/exercises.json` (**1,324 verified exercises**).
- Supports 3-Day Full Body, 4-Day Upper/Lower, and 5/6-Day Push/Pull/Legs (PPL) splits.
- Automatically selects exercises based on user experience level, equipment availability (barbell, dumbbell, machine, body weight), and target muscle group (Chest, Back, Quads, Hamstrings, Deltoids, Core).

### 2. Diet Plan Generator
- **BMR (Mifflin-St Jeor):**
  - $\text{BMR}_{\text{Male}} = 10 \cdot \text{wt} + 6.25 \cdot \text{ht} - 5 \cdot \text{age} + 5$
  - $\text{BMR}_{\text{Female}} = 10 \cdot \text{wt} + 6.25 \cdot \text{ht} - 5 \cdot \text{age} - 161$
- **TDEE & Caloric Target:** Applied physical activity multipliers and goal deficits (-500 kcal/day for fat loss, +300 kcal/day for muscle hypertrophy).
- **Macronutrients:** Protein locked at $2.0\,\text{g/kg}$ to prevent lean muscle catabolism; Fats at $25\%$; Carbohydrates filling remainder. Includes 4-meal distribution blueprint.

### 3. Dynamic 12-Week Weight Loss Simulator
- Accounts for energy physics ($7,700 \text{ kcal} \approx 1 \text{ kg fat}$) and metabolic adaptation (BMR drops by $\approx 15 \text{ kcal/day}$ per kg lost).
- Visualizes week-by-week weight trajectory and cumulative deficit curve.

---

## 📂 Project Directory Structure

```
d:\DAP Lab\Project\
├── datasets/                                 # Project Datasets & Media
│   ├── gym_members_exercise_tracking.csv     # Primary dataset (973 records)
│   ├── calories.csv                          # Secondary dataset (15,000 records)
│   └── exercises-dataset-main/               # 1,324 exercise JSON library + images/videos
├── models/                                   # Serialized trained models
│   ├── linear_regression_model.joblib
│   ├── decision_tree_model.joblib
│   ├── random_forest_model.joblib
│   ├── gradient_boosting_model.joblib
│   └── best_model_bundle.joblib              # Deployed pipeline
├── output_plots/                             # High-resolution (300 DPI) figures
│   ├── exp1_data_handling.png
│   ├── exp2_variability.png
│   ├── exp3_transformations.png
│   ├── exp4_correlation.png
│   ├── exp5_linear_regression.png
│   ├── exp6_hypothesis_testing.png
│   ├── exp8_hierarchical_analysis.png
│   ├── exp8_3d_regression_surface.png
│   ├── ml_model_comparison.png
│   └── weight_loss_trajectory.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py                        # Dataset ingest & preprocessing
│   ├── dap_lab_experiments.py                # EXP 1 to EXP 8 implementations
│   ├── ml_models.py                          # ML training & comparative eval
│   ├── calorie_predictor.py                  # Calorie prediction engine
│   └── planner_recommender.py                # Workout & diet planner
├── build_notebook.py                         # Generates Jupyter Notebook
├── fitness_analyzer_lab_project.ipynb        # Complete interactive lab notebook
├── run_experiments.py                        # Executes EXP 1 - EXP 8
├── run_ml_pipeline.py                        # Executes ML training & benchmarks
├── run_analyzer.py                           # CLI and interactive application
├── requirements.txt                          # Package dependencies
└── README.md                                 # Complete documentation
```

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All 8 DAP Lab Experiments (EXP 1 to EXP 8)
```bash
python run_experiments.py
```
*Outputs all mathematical tables and saves publication-quality plots in `output_plots/`.*

### 3. Run the Machine Learning Training & Evaluation Pipeline
```bash
python run_ml_pipeline.py
```
*Trains Linear Regression, Decision Tree, Random Forest, and Gradient Boosting models, evaluates cross-validation, and saves model bundles.*

### 4. Run the Full Application (CLI & Interactive Mode)
- **Automated End-to-End Demo:**
  ```bash
  python run_analyzer.py --demo
  ```
- **Interactive Terminal Menu:**
  ```bash
  python run_analyzer.py --interactive
  ```

### 5. Open and Run the Jupyter Notebook
Open `fitness_analyzer_lab_project.ipynb` in VS Code, PyCharm, JupyterLab, or Google Colab and run all cells sequentially.

---

## 🎓 Lab Viva Questions & Model Answers

**Q1: Why is the Geometric Mean always less than or equal to the Arithmetic Mean ($\text{AM} \ge \text{GM}$)?**  
*Answer:* By Jensen's Inequality, because the logarithm function is strictly concave, $\mathbb{E}[\ln(X)] \le \ln(\mathbb{E}[X])$. Exponentiating both sides proves $\exp(\mathbb{E}[\ln(X)]) = \text{GM} \le \text{AM}$. In our experiment, $\text{AM} = 905.42 \text{ kcal}$ and $\text{GM} = 862.11 \text{ kcal}$.

**Q2: What is the exact mathematical link between Pearson correlation $r$ and Simple Linear Regression?**  
*Answer:* In simple linear regression $\hat{y} = \beta_0 + \beta_1 x$:
1. The coefficient of determination is strictly equal to the squared Pearson correlation: $R^2 = r^2$.
2. The OLS slope is $\beta_1 = r \frac{s_y}{s_x}$, where $s_y$ and $s_x$ are sample standard deviations.

**Q3: When should Welch's t-test be preferred over Student's t-test?**  
*Answer:* Welch's t-test does not assume equal population variances ($\sigma_1^2 \neq \sigma_2^2$), adjusting the degrees of freedom using the Welch–Satterthwaite equation, which avoids inflated Type I errors.

**Q4: Why does Gradient Boosting outperform Decision Trees and Linear Regression on this dataset?**  
*Answer:* While Session Duration has a dominant linear effect, heart rate and biometric factors have non-linear synergistic interactions with duration. Gradient Boosting builds sequential shallow trees that correct previous residual errors, achieving an $R^2$ of $0.9932$ and an RMSE of only $23.88$ kcal.

