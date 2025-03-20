import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

# Create directory structure
directories = [
    'data/processed',
    'features',
    'models',
    'evaluation',
    'submissions',
    'notebooks'
]

for directory in directories:
    os.makedirs(os.path.join('/Volumes/MINT/projects/model', directory), exist_ok=True)

print("Project directory structure created successfully.")
