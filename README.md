# Network Intrusion Detection System

The project tackles the problem of detecting cyberattacks in network 
traffic automatically. Using the NSL-KDD dataset of 125,973 labeled 
network connections, it compares XGBoost and MLP neural network 
classifiers across five attack categories — DoS, Probe, R2L, U2R, 
and normal. The system extends to a federated learning simulation 
across three hospitals that share probability vectors instead of raw 
data, losing only 1% accuracy while preserving complete data privacy. 
Differential privacy via Laplace noise is added to quantify the 
privacy-accuracy tradeoff.

## What it implements

- **detect.py** — loads and preprocesses NSL-KDD, maps 22 attack 
  labels to 5 categories, trains XGBoost baseline achieving 77% 
  accuracy with DoS F1=0.88 but R2L F1=0.09 due to class imbalance.

- **mlp_model.py** — trains a 41→128→64→5 MLP neural network on 
  the same data. Achieves 99.5% training accuracy but 74% test 
  accuracy — underperforms XGBoost on minority classes, demonstrating 
  that model complexity does not always beat simpler approaches.

- **federated.py** — simulates federated learning across 3 hospital 
  clients. Each trains locally on one third of data, shares only 
  probability vectors via soft-voting aggregation. Federated accuracy 
  76% vs centralised 77% — 1% privacy cost.

- **differential_privacy.py** — adds Laplace noise to probability 
  vectors before sharing. Sweeps epsilon from 0.1 to 10, measuring 
  the privacy-accuracy tradeoff. Sweet spot at epsilon=2: medium 
  privacy with Macro F1=0.45 and DoS F1=0.78.

## Key results

| Model | Accuracy | DoS F1 | R2L F1 | U2R F1 |
|---|---|---|---|---|
| XGBoost centralised | 77% | 0.88 | 0.09 | 0.08 |
| MLP neural network | 74% | 0.84 | 0.04 | 0.13 |
| Federated XGBoost | 76% | 0.88 | 0.08 | 0.05 |
| Federated + DP (ε=2) | ~72% | 0.78 | 0.05 | ~0.03 |

**Key insight:** R2L and U2R recall remains poor across all models 
due to severe class imbalance — 52 U2R samples vs 67,343 normal. 
This is a known limitation of NSL-KDD and an active research problem.

## Privacy-accuracy tradeoff

| Epsilon | Macro F1 | DoS F1 | Privacy level |
|---|---|---|---|
| 0.1 | 0.18 | 0.34 | Strong |
| 2.0 | 0.45 | 0.78 | Medium |
| 10.0 | 0.51 | 0.88 | Weak |

## How to run

```bash
pip install pandas scikit-learn xgboost torch matplotlib seaborn
git clone https://github.com/ReggieJOE/network-intrusion-detection.git
cd network-intrusion-detection
# Download KDDTrain+.txt and KDDTest+.txt manually into the folder
python detect.py
python mlp_model.py
python federated.py
python differential_privacy.py
```

## Research connection

This project connects to Prof. Dong Lin's (Illinois Tech) research 
on ML-based IoT botnet detection with differential privacy — 
specifically the federated detection architecture and the epsilon 
sweep measuring the privacy-accuracy tradeoff. The finding that 
federated soft-voting loses only 1% accuracy vs centralised training 
supports the viability of privacy-preserving IDS in real healthcare 
networks. Prof. Walid Saad's (Virginia Tech) work on federated 
learning for edge networks provides the theoretical foundation for 
extending this approach to distributed IoT environments.

## Author

Reginald Jojo Gwira  
Kwame Nkrumah University of Science and Technology, Ghana  
GitHub: [ReggieJOE](https://github.com/ReggieJOE)