import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- 1) Embedding ve veri yükleme ---
print("Veri yükleniyor...")
X = joblib.load("Tweets_BERT_embeddings.joblib")  # [N, 768]
df = pd.read_csv("Tweets_clean.csv")
y = df["airline_sentiment"]  # label kolonu

# --- 2) Train-test bölme ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3) Linear SVM ---
print("\n--- Linear SVM ---")
svm_clf = LinearSVC(random_state=42)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
print("Doğruluk Oranı (SVM):", accuracy_score(y_test, y_pred_svm))
print("Classification Report (SVM):\n", classification_report(y_test, y_pred_svm))
joblib.dump(svm_clf, "Tweets_SVM_model.joblib")

# --- 4) Logistic Regression ---
print("\n--- Logistic Regression ---")
lr_clf = LogisticRegression(max_iter=1000, solver="lbfgs")
lr_clf.fit(X_train, y_train)
y_pred_lr = lr_clf.predict(X_test)
print("Doğruluk Oranı (LR):", accuracy_score(y_test, y_pred_lr))
print("Classification Report (LR):\n", classification_report(y_test, y_pred_lr))
joblib.dump(lr_clf, "Tweets_LR_model.joblib")

# --- 5) MLP ---
print("\n--- Multi-Layer Perceptron (MLP) ---")
mlp_clf = MLPClassifier(hidden_layer_sizes=(256,128), 
                        activation="relu", 
                        solver="adam", 
                        max_iter=100, 
                        random_state=42, 
                        verbose=True)

mlp_clf.fit(X_train, y_train)
y_pred_mlp = mlp_clf.predict(X_test)
print("Doğruluk Oranı (MLP):", accuracy_score(y_test, y_pred_mlp))
print("Classification Report (MLP):\n", classification_report(y_test, y_pred_mlp))
joblib.dump(mlp_clf, "Tweets_MLP_model.joblib")





# import joblib
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score


# # Embedding ve veri yükleme
# print("Veri yükleniyor...")
# X = joblib.load("Tweets_BERT_embeddings.joblib")  # [N, 768]
# df = pd.read_csv("Tweets_clean.csv")
# y = df["airline_sentiment"]  # label kolonu


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )


# # Random Forest Model

# print("\n--- Random Forest ---")
# rf_clf = RandomForestClassifier(
#     n_estimators=200,   # ağaç sayısı
#     max_depth=None,     # derinlik sınırlaması yok
#     random_state=42,
#     n_jobs=-1           # CPU'nun tüm çekirdeklerini kullan
# )

# # Eğitme
# rf_clf.fit(X_train, y_train)

# # Tahmin
# y_pred_rf = rf_clf.predict(X_test)

# # Değerlendirme
# print("Doğruluk Oranı (RF):", accuracy_score(y_test, y_pred_rf))
# print("Classification Report (RF):\n", classification_report(y_test, y_pred_rf))

# # Modeli kaydetme
# joblib.dump(rf_clf, "Tweets_RF_model.joblib")
# print("Random Forest modeli 'Tweets_RF_model.joblib' olarak kaydedildi.")
