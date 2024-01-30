from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

data = [
    ("I love this sandwich", "pos"),
    ("This is an amazing place", "pos"),
    ("I feel very good about these cheese", "neutral"),
    ("This is my best work", "pos"),
    ("What an awesome view", "pos"),
    ("I do not like this restaurant", "neg"),
    ("I am tired of this stuff", "neg"),
    ("I can’t deal with this", "neg"),
    ("He is my sworn enemy", "neg"),
    ("My boss is horrible", "neg"),
    ("This is an awesome place", "pos"),
    ("I do not like the taste of this juice", "neg"),
    ("I love to dance", "pos"),
    ("I am sick and tired of this place", "neg"),
    ("What a great holiday", "pos"),
    ("That is a bad locality to stay", "neg"),
    ("We will have good fun tomorrow", "pos"),
    ("I went to my enemy's house today", "neg")
]

texts, labels = zip(*data)
binary_labels = [1 if label == "pos" else 0 for label in labels]
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, binary_labels, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)

classifier = MultinomialNB()
classifier.fit(X_train, labels_train)
predictions = classifier.predict(X_test)

total_instances = len(texts_test)
accuracy = accuracy_score(labels_test, predictions)
recall = recall_score(labels_test, predictions)
precision = precision_score(labels_test, predictions)
conf_matrix = confusion_matrix(labels_test, predictions)

print("1. Total Instances of Dataset:", len(data))
print("2. Accuracy:", accuracy)
print("3. Recall:", recall)
print("4. Precision:", precision)
print("5. Confusion Matrix:")
print(conf_matrix)
