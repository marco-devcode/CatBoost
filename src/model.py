# src/model.py
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

# Carica i dati preparati nella Fase 3
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',   # 3 classi: H / D / A
    eval_metric='Accuracy',
    random_seed=42,
    verbose=100                   # stampa ogni 100 iterazioni
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50      # ferma se non migliora per 50 round
)