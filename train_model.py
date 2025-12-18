import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("language.csv")

X = data["Text"]
y = data["language"]

# Vectorization
cv = CountVectorizer()
X_vec = cv.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save vectorizer & model
pickle.dump(cv, open("cv.pkl", "wb"))
pickle.dump(model, open("language_model.pkl", "wb"))

print("✅ cv.pkl and language_model.pkl created successfully")
