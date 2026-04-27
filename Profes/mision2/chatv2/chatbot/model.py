import os # Trabaja con el sistema operativo
import pickle# Guarda y carga objetos de python en archivos
from sklearn.feature_extraction.text import CountVectorizer
# CountVectorizer convierte texto en un vector
from sklearn.naive_bayes import MultinomialNB
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR,"model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR,"vectorizer.pkl")
ANSWERS_PATH = os.path.join(MODEL_DIR,"answers.pkl")

def buid_and_train_model(train_pairs):
    questions = [q for q, _ in train_pairs]  
    answers = [a for _, a in train_pairs]
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(questions)
    unique_answers = sorted(set(answers))
    answer_to_label ={a: i for i, a in enumerate(unique_answers)}
    y = [answer_to_label[a] for a in answers]




