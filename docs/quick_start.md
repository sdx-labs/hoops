# Basketball Prediction System - Quick Start Guide

This guide will help you get up and running with the basketball tournament prediction system.

## Prerequisites

- Python 3.10 or higher
- Kaggle API credentials (for downloading competition data)
- Git (for version control)

## Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd hoops
```

2. **Set up your Kaggle API credentials**

Create a file at `~/.kaggle/kaggle.json` with your API credentials:
```json
{
  "username": "your-username",
  "key": "your-key"
}
```

3. **Set up your Python environment**

Using a virtual environment is recommended:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Download Competition Data

```bash
python src/data/download.py
```

## Run the Complete Pipeline

```bash
python src/run_pipeline.py
```

This will:
1. Preprocess the data
2. Generate features
3. Train models
4. Generate predictions

## Generate Predictions Only

If you already have trained models and just want to generate predictions:

```bash
python src/models/predict_model.py
```

For ensemble predictions:

```bash
python src/models/predict_model.py --ensemble
```

## Viewing Results

After running the pipeline, you'll find:

- Feature analysis in `data/validation/`
- Models in `models/`
- Predictions in `data/predictions/`

## Advanced Usage

See the full documentation for more advanced usage options and customization.
