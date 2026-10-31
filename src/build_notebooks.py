import json
from pathlib import Path

notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(parents=True, exist_ok=True)

def create_nb(cells, filename):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(notebooks_dir / filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebooks/{filename}")

# 1. 01_eda.ipynb
cells_eda = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 01. Exploratory Data Analysis & Anomaly Signature Profiling\n",
            "**Project**: Solar & Electrical Power System Fault and Anomaly Detection Using Stacking Ensemble Learning\n",
            "**Dataset**: A Real Controlled-Fault Benchmark for Power-Anomaly Detection (Zenodo CC BY 4.0)\n",
            "**Total Scale**: 1,814,400 per-second records across 3 IoT Edge monitoring stations."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from src.config import STATION_FILES, RAW_SENSOR_FEATURES, CATEGORY_LABELS, SEVERITY_LABELS\n",
            "\n",
            "print('Configurations and dependencies successfully loaded!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Data Sample from Station 1 (Laboratory Baseline)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_sample = pd.read_csv(STATION_FILES[1], nrows=50000)\n",
            "print('Sample shape:', df_sample.shape)\n",
            "df_sample[['timestamp', 'voltage_V', 'current_A', 'power_W', 'frequency_Hz', 'power_factor', 'fault_category', 'severity']].head(10)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Fault Category & Severity Distributions"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
            "df_sample['fault_category'].value_counts().plot(kind='bar', ax=axes[0], color='royalblue', title='Fault Categories')\n",
            "axes[0].set_ylabel('Sample Count')\n",
            "axes[0].grid(axis='y', alpha=0.3)\n",
            "\n",
            "df_sample['severity'].value_counts().plot(kind='bar', ax=axes[1], color='crimson', title='Fault Severity Grades')\n",
            "axes[1].set_ylabel('Sample Count')\n",
            "axes[1].grid(axis='y', alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Physical Sensor Telemetry Correlation Matrix"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(9, 7))\n",
            "corr = df_sample[RAW_SENSOR_FEATURES].corr()\n",
            "sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)\n",
            "plt.title('Electrical Telemetry Correlation Heatmap')\n",
            "plt.show()"
        ]
    }
]
create_nb(cells_eda, "01_eda.ipynb")

# 2. 02_preprocessing_and_splitting.ipynb
cells_prep = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 02. Preprocessing, Feature Engineering & Group-Aware Splitting\n",
            "Ensures **zero data leakage** by keeping complete fault event durations within single splits."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "from src.preprocessing import build_curated_benchmark_dataset, create_leakage_free_splits\n",
            "\n",
            "# Loads all 30k fault samples and balanced normal baseline (~75k total records)\n",
            "df_curated = build_curated_benchmark_dataset(sample_normal_per_station=15000)\n",
            "train_df, test_df = create_leakage_free_splits(df_curated)\n",
            "\n",
            "print(f'Train partition: {len(train_df):,} rows')\n",
            "print(f'Test partition:  {len(test_df):,} rows')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Verify Zero Event Leakage"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "train_events = set(train_df[train_df['event_id'] != 0]['group_id'].unique())\n",
            "test_events = set(test_df[test_df['event_id'] != 0]['group_id'].unique())\n",
            "overlap = train_events.intersection(test_events)\n",
            "\n",
            "print('Train distinct fault events:', len(train_events))\n",
            "print('Test distinct fault events:', len(test_events))\n",
            "print('Overlapping event count (MUST BE 0):', len(overlap))\n",
            "assert len(overlap) == 0, 'Data leakage detected across train and test!'"
        ]
    }
]
create_nb(cells_prep, "02_preprocessing_and_splitting.ipynb")

# 3. 03_model_training_and_stacking.ipynb
cells_train = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 03. Model Training & Stacking Ensemble Learning\n",
            "Train 5 Base Classifiers:\n",
            "1. Random Forest\n",
            "2. Extra Trees\n",
            "3. Gradient Boosting\n",
            "4. K-Nearest Neighbors\n",
            "5. Linear Support Vector Machine\n",
            "\n",
            "And combine them via a **Logistic Regression Meta-Model (Stacking)** + **Soft / Hard Voting**."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import joblib\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "from src.config import PROCESSED_DATA_DIR, MODELS_DIR, FEATURE_COLS, CATEGORY_LABELS, SEVERITY_LABELS\n",
            "from src.train import train_and_stack_pipeline\n",
            "\n",
            "print('Ready to train models manually on your local system!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Train Fault Category Pipeline (8 Classes)\n",
            "Generates 5-fold Group-Aware OOF predictions and trains base models + Stacking meta-model."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Execute this cell when you want to run Fault Category training\n",
            "train_and_stack_pipeline(target_col='target_category_idx', model_subfolder='fault_category')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Train Fault Severity Pipeline (5 Grades)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Execute this cell when you want to run Fault Severity training\n",
            "train_and_stack_pipeline(target_col='target_severity_idx', model_subfolder='fault_severity')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Compare Models & Display Evaluation Metrics"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from src.evaluate import evaluate_all_models\n",
            "results_category = evaluate_all_models(target_col='target_category_idx', model_subfolder='fault_category')\n",
            "results_category"
        ]
    }
]
create_nb(cells_train, "03_model_training_and_stacking.ipynb")
