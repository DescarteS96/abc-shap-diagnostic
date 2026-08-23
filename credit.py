# credit.py — version alternative
from ucimlrepo import fetch_ucirepo
import pandas as pd

dataset = fetch_ucirepo(id=350)
X = dataset.data.features
y = dataset.data.targets
df = pd.concat([X, y], axis=1)
df.to_csv('credit_default.csv', index=False)
print("Exporté :", df.shape)