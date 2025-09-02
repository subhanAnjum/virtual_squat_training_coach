# AI Squat Coach

This project provides a comprehensive machine learning pipeline to analyze squat exercises from video. It leverages MediaPipe BlazePose for pose estimation and a custom-trained model to classify squat quality, count repetitions, and provide actionable feedback.

The core of this project is a Jupyter Notebook (`AI_Coach_Old_Approach_(Auto_Labeling).ipynb`) that demonstrates an end-to-end workflow: data acquisition, automated labeling, model training, and evaluation.

## Features

-   **Pose Estimation:** Utilizes MediaPipe BlazePose to extract 33 full-body 3D landmarks.
-   **Automated Data Labeling:** A rule-based engine processes videos to automatically label each squat repetition into one of four classes:
    -   `good`: A well-executed squat meeting all biomechanical criteria.
    -   `too_low`: The squat depth is excessive, potentially leading to "butt wink."
    -   `bad_posture`: Indicates issues like excessive forward torso lean or knee valgus (knees caving inward).
-   **Robust Model Training:** Implements a best-practice training pipeline including:
    -   **SMOTE:** To handle class imbalance in the auto-labeled dataset.
    -   **Hyperparameter Tuning:** Uses `GridSearchCV` to find the optimal parameters for multiple models.
    -   **Model Comparison:** Trains and evaluates both `RandomForest` and `XGBoost` classifiers to select the best performer.
-   **Comprehensive Evaluation:** Provides a full suite of evaluation metrics, including accuracy, classification reports, a confusion matrix, feature importance plots, and multi-class ROC curves.
-   **Real-time Inference:** A `run.py` script allows for real-time coaching using a webcam.

## Project Structure

```
.
├── ai_coach/               # Main Python package for the AI coach logic
│   ├── exercises.py        # Squat-specific logic, constants, and the rule-based SquatAdvisor
│   ├── per_rep_features.py # Logic for aggregating features over a single repetition
│   └── ...                 # Other modules for pose, features, camera, etc.
├── data/
│   └── youtube_videos/     # Directory for downloaded YouTube videos
├── models/                 # Saved machine learning models (e.g., squat_quality_multiclass_auto.joblib)
├── outputs/                # Output directory for processed videos and evaluation artifacts
├── test_video/             # Sample videos for testing the inference pipeline
├── AI_Coach_Old_Approach_(Auto_Labeling).ipynb  # Main notebook for training and evaluation
├── requirements.txt        # Project dependencies
├── run.py                  # Script for real-time coaching demo
└── README.md
```

## Setup

This project uses Python 3.10+ and requires several packages.

1.  **Clone the repository and navigate to the project root.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *Note: On macOS, you may need to install `libomp` for XGBoost to work correctly: `brew install libomp`.*

## Usage

The primary workflow is managed through the Jupyter Notebook.

### 1. Training and Evaluation

1.  Launch Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    ```
2.  Open `AI_Coach_Old_Approach_(Auto_Labeling).ipynb`.
3.  Run the cells sequentially from top to bottom. The notebook will:
    -   Download the required Kaggle dataset and sample YouTube videos.
    -   Process all videos to extract and auto-label squat repetitions.
    -   Train, evaluate, and save the best squat quality classifier to the `models/` directory.
    -   Run inference on the videos in the `test_video/` folder and save the annotated output to `outputs/`.

### 2. Real-time Demo

Once a model has been trained and saved by the notebook, you can run the real-time coach.

-   **To use your default webcam:**
    ```bash
    python run.py realtime
    ```
-   **To specify a camera by index:**
    ```bash
    python run.py realtime --index 0
    ```

The application will display your camera feed with pose landmarks, a rep counter, and real-time feedback on your squat form. 