# Federated Learning for Privacy-Preserving Fraud Detection

A proof-of-concept system demonstrating how multiple parties (e.g., banks) can collaboratively train a powerful fraud detection model using Federated Learning (FL) and Differential Privacy (DP), without sharing their sensitive raw data. This project explores the challenges and trade-offs of applying these privacy-enhancing technologies in a realistic, highly imbalanced, and non-identically distributed data environment.

<p align="center">
  <img src="https://flower.ai/docs/framework/_images/flower-architecture-basic-architecture.svg" alt="Federated Learning Architecture">
</p>
<p align="center">Image courtesy of the Flower Framework</p>

## Table of Contents
- [Problem Statement](#problem-statement)
- [Solution: A Privacy-First ML Approach](#solution-a-privacy-first-ml-approach)
- [Key Features](#key-features)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run Simulations](#how-to-run-simulations)
- [Results and Analysis](#results-and-analysis)
- [Conclusion](#conclusion)
- [Potential Extensions & Future Work](#potential-extensions--future-work)

## Problem Statement
In the financial sector, data is both a critical asset and a significant liability. Strict data privacy regulations, such as the GDPR and CCPA, impose heavy fines for data breaches and mandate stringent data handling practices. These laws create data silos, where individual banks possess valuable transaction data but are legally prevented from sharing it with others.

This presents a major obstacle for building robust machine learning models. A fraud detection model trained on data from a single bank is limited by that bank's specific data patterns. A model trained on a diverse dataset from multiple banks would be far more effective at identifying new and complex fraud schemes. However, data sharing is not an option.

This project directly addresses this market pain: How can competing institutions collaboratively build a superior fraud detection model without violating data privacy laws and exposing sensitive customer information?

## Solution: A Privacy-First ML Approach
This project leverages two cutting-edge technologies to turn privacy from a barrier into an enabler:

1. **Federated Learning (FL)**: Instead of pooling data, FL allows multiple clients (banks) to train a model locally on their own private data. They then send only the model updates (weights and gradients) to a central server. The server aggregates these updates to create an improved global model, which is then sent back to the clients for the next round of training. At no point does raw data leave the client's premises.

2. **Differential Privacy (DP)**: To add an even stronger, mathematically provable layer of privacy, we integrate Differential Privacy using the Opacus library. DP adds carefully calibrated statistical noise during the client-side training process. This ensures that the contribution of any single transaction is masked, making it computationally infeasible to determine if a specific individual's data was part of the training set, even if an attacker could inspect the model updates.

## Key Features
- **Federated Learning Orchestration**: Uses the Flower framework to simulate a multi-client FL environment.
- **Differential Privacy Integration**: Employs Opacus to train PyTorch models with differential privacy guarantees.
- **Realistic Data Simulation**: Simulates Non-IID data across clients using a Dirichlet distribution.
- **Imbalanced Data Handling**: Implements client-side weighted loss.
- **Comprehensive Benchmarking**: Includes centralized training for performance comparison.
- **Detailed Evaluation**: Measures Precision, Recall, and F1-score.

## Technical Stack
- ML Framework: PyTorch
- Federated Learning: Flower (flwr)
- Differential Privacy: Opacus
- Data Manipulation: Pandas, NumPy
- ML Utilities: scikit-learn

## Project Structure
```
.
├── Data/
│   └── creditcard.csv            # Placeholder for the dataset
├── Data_Load_and_Prep/
│   ├── __init__.py
│   └── data_utils.py             # Data loading and Non-IID partitioning logic
├── Model/
│   ├── __init__.py
│   └── model.py                  # PyTorch model definition (FraudNet)
├── client.py                     # Flower client logic with DP and weighted loss
├── run_simulation.py             # Main script for FL simulations
├── train_centralized.py          # Centralized benchmark training script
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## Setup and Installation

### 1. Prerequisites
- Python 3.8 or higher
- git for cloning the repository

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd federated-fraud-detection
```

### 3. Set Up a Virtual Environment
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download the Dataset
1. Download the "Credit Card Fraud Detection" dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in the `Data/` directory

## How to Run Simulations

### 1. The Centralized Benchmark
```bash
python train_centralized.py
```

### 2. Federated Learning Simulations

#### Experiment A: Non-Private FL with Non-IID Data
In `run_simulation.py`, set:
```python
NON_IID_ALPHA = 5.0
DP_ENABLED = False
```

Run:
```bash
python run_simulation.py
```

#### Experiment B: Differentially Private FL with Non-IID Data
In `run_simulation.py`, set:
```python
NON_IID_ALPHA = 5.0
DP_ENABLED = True
DP_TARGET_EPSILON = 10.0
DP_MAX_GRAD_NORM = 1.0
LEARNING_RATE = 0.0005
```

Run:
```bash
python run_simulation.py
```

## Results and Analysis

### Performance Summary Table

| Training Method | Data Setting (Non-IID α) | Privacy (ε) | Accuracy | Precision | Recall | F1-Score | Key Observation |
|----------------|-------------------------|-------------|----------|-----------|---------|-----------|-----------------|
| Centralized    | N/A                    | None        | 0.9993   | 0.8125    | 0.7959  | 0.8041    | Performance Benchmark |
| Federated      | 0.5 (Extreme)          | None        | 0.9983   | 0.0000    | 0.0000  | 0.0000    | Model fails; extreme Non-IID is the bottleneck |
| Federated      | 5.0 (Moderate)         | None        | 0.9994   | 0.7961    | 0.8367  | 0.8159    | Success! Comparable to centralized |
| Federated      | 5.0 (Moderate)         | 50.0        | 0.9992   | 0.8293    | 0.6939  | 0.7556    | Good precision but recall suffers |
| Federated      | 5.0 (Moderate)         | 10.0        | 0.9983   | 0.0000    | 0.0000  | 0.0000    | Model fails; noise too high |

### Key Insights
1. **Extreme Non-IID is the Primary Challenge**: Data heterogeneity is a fundamental obstacle.
2. **Mitigating Non-IID is Key**: FedAvg can succeed with manageable data distribution.
3. **The Privacy-Utility Trade-off is Real**: Stronger privacy (ε ≤ 10) significantly impacts model performance.

## Conclusion
This project demonstrates successful implementation of FL with DP for imbalanced fraud detection. The key challenge lies in balancing data heterogeneity with privacy requirements. While FL can match centralized performance, achieving strong privacy guarantees while maintaining high utility requires advanced techniques.

## Potential Extensions & Future Work
1. **Advanced DP Tuning**: Optimize privacy parameters for better utility
2. **More Advanced FL Algorithms**: Test FedProx, SCAFFOLD, or FedNova
3. **Modern Flower App Structure**: Refactor using current best practices
4. **Client-Side Fine-tuning**: Explore personalization techniques
5. **Secure Aggregation**: Add additional privacy layers
