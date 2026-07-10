# Mg-Ni-ML
Machine-learning workflow, formatted dataset, and publication figures for Mg-Ni hydrogen-storage alloy property prediction and candidate screening.

This repository accompanies the manuscript:

**Integrated nine-model machine learning, ablation analysis and uncertainty-guided active learning for Mg-Ni hydrogen-storage alloys**

The project builds a reproducible Mg-Ni-focused machine-learning pipeline for three coupled hydrogen-storage properties:

- hydrogen storage capacity, `H_storage_wt%`
- absorption pressure, `Pressure_bar`
- absorption temperature, `Temperature_K`

The final formatted dataset contains **5888 valid Mg-Ni-containing entries** extracted and integrated from public augmented Mg-based alloy data. The workflow combines descriptor construction, nine-model regression, hyperparameter optimization, feature ablation, SHAP interpretation, and uncertainty-guided UCB candidate screening.

## Repository Contents

```text
Mg-Ni-ML/
|-- input_data/
|   `-- MgNi_Augmented_Data_Format.xlsx    # Final formatted Mg-Ni dataset
`-- src/
    `-- MgNi_Integrated_ML.py              # Final ML, plotting, ablation and UCB-screening code
```

## Methods

The main script benchmarks nine regression models:

1. Ridge linear regression
2. Support vector regression
3. Decision tree
4. Random forest
5. Gradient boosting regression tree
6. XGBoost
7. ExtraTrees
8. CatBoost
9. LightGBM

For each target property, the code constructs target-specific feature sets from Mg-Ni alloy composition, engineered physical descriptors, and process variables. It then performs model comparison, feature ablation, SHAP-based model interpretation, and UCB-based virtual candidate screening.

## Main Results

| Target property | Best model | Test R2 | Notes |
|---|---:|---:|---|
| Hydrogen storage capacity | LightGBM | 0.9611 | RMSE = 0.2520 wt%, MAE = 0.1550 wt%, MAPE = 4.6614% |
| Absorption pressure | XGBoost | 0.8595 | Best pressure model in the nine-model benchmark |
| Absorption temperature | XGBoost | 0.8618 | Best temperature model in the nine-model benchmark |

Feature ablation shows that both physical descriptors and process variables are important for robust prediction. The uncertainty-guided UCB screen prioritizes Mg-rich, La-containing candidate alloys for future experimental validation; for example, `Mg93.32Ni3.83La2.85` is ranked as a representative high-capacity candidate with a predicted capacity of about `6.80 wt%`.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/calm0001/Mg-Ni-ML.git
cd Mg-Ni-ML
python -m pip install pandas numpy scipy scikit-learn matplotlib seaborn pillow openpyxl xgboost lightgbm catboost shap
```

Core dependencies include `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`, `catboost`, `shap`, `pillow`, `python-docx`, and `openpyxl`.

## Quick Use

### 1. Re-run the complete workflow

The full workflow retrains the models, regenerates predictions, runs ablation analysis, performs UCB candidate screening, and exports publication-style figures.

```bash
python src/MgNi_Integrated_ML.py ^
  --data input_data/MgNi_Augmented_Data_Format.xlsx ^
  --output-dir outputs/integrated_ml ^
  --targets all
```

For a quick smoke test:

```bash
python src/MgNi_Integrated_ML.py ^
  --data input_data/MgNi_Augmented_Data_Format.xlsx ^
  --output-dir outputs/quick_test ^
  --targets all ^
  --quick
```

### 2. Regenerate publication-style figures

After a complete run has generated the required model outputs, publication-style figures can be regenerated from the same output directory.

```bash
python src/MgNi_Integrated_ML.py ^
  --data input_data/MgNi_Augmented_Data_Format.xlsx ^
  --output-dir outputs/integrated_ml ^
  --only-publication-outputs
```

## Outputs

The workflow generates:

- model-performance tables for all nine regressors
- prediction files for training and testing samples
- target-specific processed feature tables
- feature-ablation summaries
- SHAP bee swarm plots for the best models
- UCB-screened candidate-composition tables
- publication figures corresponding to Fig. 1-Fig. 10 in the manuscript

Important generated files include:

```text
outputs/integrated_ml/summary_best_models.csv
outputs/integrated_ml/*/model_comparison_*.csv
outputs/integrated_ml/*/ablation_results_*.csv
outputs/integrated_ml/capacity/active_learning_screened_candidates_capacity.csv
outputs/integrated_ml/MSEB_submission_figures/
```

## Notes on Reproducibility

The model split uses a fixed random seed of `42`. Small numerical differences may still occur across systems because of changes in package versions, parallel tree-building behavior, or optional model backends such as XGBoost, CatBoost and LightGBM.

The active-learning section is an uncertainty-guided virtual screening procedure. The prioritized compositions should be interpreted as model-suggested candidates for experimental validation rather than confirmed materials.

## Citation

If this repository is useful for your work, please cite the associated manuscript after publication. Before publication, cite this repository as:

```text
Cai, J.; Lai, Q.; Xie, Y.; Wei, X.; Luo, X.; Zhao, X.; Xiao, C.; Yang, J.
Mg-Ni-ML: Integrated nine-model machine learning, ablation analysis and uncertainty-guided active learning for Mg-Ni hydrogen-storage alloys.
GitHub repository, 2026.
https://github.com/calm0001/Mg-Ni-ML
```

## Contact

Corresponding author: **Qi Lai**  
E-mail: `49129834@qq.com`
