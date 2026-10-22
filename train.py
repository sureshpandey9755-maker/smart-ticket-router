import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
import joblib

# 1. Data load karein
df = pd.read_csv("data/tickets.csv")

# 2. Train/Test split (stratify ke sath)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["department"], test_size=0.2, random_state=42, stratify=df["department"]
)

# 3. TF-IDF Vectorization
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 4. Multiple models define karein
models = {
    "Naive Bayes": MultinomialNB(alpha=0.5),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced"),
}

results = []
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    results.append((name, acc, f1))

# Comparison table print karein
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Macro F1"])
print("\n--- Model Comparison ---")
print(results_df)
import joblib
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Best model ko select karke save karein (jaise Logistic Regression ya Linear SVM)
best_model = models["Logistic Regression"]

# Model aur vectorizer ko dictionary mein save karein
joblib.dump({
    "dept": best_model,
    "tfidf": tfidf
}, "models/ticket_model.joblib")

print("\nModel successfully saved to models/ticket_model.joblib!")