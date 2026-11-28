# MP241 — Tea Leaf Plucking Prediction

## Group Information
- **Group ID:** G129
- **Problem ID:** MP241
- **Problem:** Tea Leaf Plucking — Guwahati
- **Faculty:** Ravi Pratap
- **Batch:** 13

## Objective
Predict the quantity of tea leaves plucked by a worker in a day using machine learning regression.

**Target:** `kg_per_worker`

## Dataset
- `train.csv` — Training data
- `later.csv` — Later data used for drift monitoring and retraining
- `holdout_features.csv` — Final prediction dataset

Training datasets are versioned using DVC.

### Dataset Features
The dataset contains information about tea workers, weather conditions,
terrain, tea bushes, and working conditions.

Key features include:

- Worker experience
- Weekly rainfall
- Slope of the terrain
- Temperature
- Bush age
- Hours worked
- Tea flush
- Garden section

The model predicts `kg_per_worker`, representing the quantity of tea
leaves plucked per worker.

## Project Pipeline
1. Data versioning with DVC
2. Data quality investigation and cleaning
3. ML experiments with MLflow
4. Model registration
5. FastAPI prediction service
6. Docker containerization
7. GitHub Actions CI
8. Kubernetes deployment
9. Drift monitoring and retraining

## Technologies
Python, Pandas, Scikit-learn, DVC, MLflow, FastAPI, Docker, Kubernetes, GitHub Actions
