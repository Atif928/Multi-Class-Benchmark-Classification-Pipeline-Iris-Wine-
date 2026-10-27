# Datheon Machine Learning & Data Science Internship

This workspace contains the machine learning projects completed during the Datheon
Machine Learning & Data Science Internship.

## Projects

| Task | Project | Documentation |
| ---- | ------- | ------------- |
| Task 1 | Multi-Class Benchmark Classification Pipeline (Iris & Wine) | [Task 1 repository](https://github.com/Atif928/Multi-Class-Benchmark-Classification-Pipeline-Iris-Wine-) |
| Task 2 | Real Estate Housing Price Prediction (OLS, Ridge & Lasso) | [Task 2 README](task_2_house_price_prediction/README.md) |

Task 1 is maintained in its original, separate repository and directory. It was not
modified as part of Task 2.

## Task 2 at a glance

Task 2 predicts median California house values with Ordinary Least Squares, Ridge
(L2), and Lasso (L1) regression. It includes a reusable `src/` package, a fully
executed Jupyter notebook, pytest tests, exploratory and diagnostic figures, and
reproducible reports.

```bash
cd task_2_house_price_prediction
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
pytest
```

Full setup, usage, results, and limitations are documented in the
[Task 2 README](task_2_house_price_prediction/README.md).
