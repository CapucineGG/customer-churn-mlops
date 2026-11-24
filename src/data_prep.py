import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Chargement du dataset
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Analyse exploratoire
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.head())

for col in df.select_dtypes(include='str').columns:
    if col != 'customerID':
        print(f"{col} : {df[col].unique()}")

print(df['Churn'].value_counts(normalize=True))
df['Churn'].value_counts(normalize=True).plot(kind='bar')
plt.title('Distribution de la target')
plt.show()

df.hist(figsize=(16, 12), bins=30)
plt.tight_layout()
plt.show()

num_cols = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(14, 10))
sns.heatmap(num_cols.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.show()

for col in num_cols.columns:
    df.boxplot(column=col, by='Churn', figsize=(8, 5))
    plt.title(f'{col} selon Churn')
    plt.suptitle('')
    plt.show()

# 3. Valeurs manquantes
# TotalCharges est lu comme du texte : les valeurs vides (" ") deviennent NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(df.isnull().sum())

null_rates = df.isnull().mean() * 100
null_rates[null_rates > 0].sort_values().plot(kind='barh', figsize=(8, 6))
plt.title('Taux de valeurs manquantes (%)')
plt.xlabel('%')
plt.show()

# Valeurs manquantes rares : suppression des lignes
df = df.dropna()
print(df.shape)

# 4. Variables catégorielles
df = df.drop(columns=['customerID'])

cat_cols = [col for col in df.select_dtypes(include='str').columns if col != 'Churn']
binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
multi_cols = [col for col in cat_cols if df[col].nunique() > 2]

le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
df = pd.get_dummies(df, columns=multi_cols)

# 5. Encodage de la variable cible (No : 0, Yes : 1)
df['Churn'] = le.fit_transform(df['Churn'])
print(df['Churn'].value_counts())

# 6. Préparation des ensembles train et test
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7. Normalisation
# fit sur train uniquement pour éviter le data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train : {X_train_scaled.shape}  X_test : {X_test_scaled.shape}")
print(f"Taux de churn train : {y_train.mean():.3f}  test : {y_test.mean():.3f}")