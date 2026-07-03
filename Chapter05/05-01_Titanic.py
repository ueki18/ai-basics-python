import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# -------------------------------------------------------
# データの読み込みと確認
# -------------------------------------------------------
df = sns.load_dataset('titanic')
print(df.shape)    # 行数・列数の確認
print(df.head())   # 先頭5行の表示
print(df.dtypes)   # 各列のデータ型

# -------------------------------------------------------
# 不要な列の削除
# -------------------------------------------------------
drop_cols = ['class', 'who', 'adult_male',
             'embark_town', 'alive', 'alone', 'deck']
df = df.drop(columns=drop_cols)

# -------------------------------------------------------
# 欠損値の確認と処理
# -------------------------------------------------------
print(df.isnull().sum())

# age：中央値で補完
df['age'] = df['age'].fillna(df['age'].median())

# embarked：最頻値で補完
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# 処理後の確認（すべて0になっていればOK）
print(df.isnull().sum())

# -------------------------------------------------------
# カテゴリ変数の数値変換
# -------------------------------------------------------
# sex：male→0，female→1 に変換
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# embarked：ワンホットエンコーディング
df = pd.get_dummies(df, columns=['embarked'], drop_first=True, dtype=int)

print(df.head())

# -------------------------------------------------------
# モデルの設計・学習・評価
# -------------------------------------------------------
# 特徴量と目的変数に分割
X = df.drop(columns=['survived'])
y = df['survived']

# 学習データとテストデータに分割（学習80%・テスト20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# モデルの学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 予測と評価
y_pred = model.predict(X_test)
print("Accuracy  :", accuracy_score(y_test, y_pred))
print("F1-score  :", f1_score(y_test, y_pred))
print("混同行列:\n", confusion_matrix(y_test, y_pred))
