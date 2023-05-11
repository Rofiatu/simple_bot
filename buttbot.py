import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import time
import datetime
import pandas as pd

nltk.download('punkt')
nltk.download('wordnet')

# Load the text file and preprocess the data
with open('text_files/buttbot.txt', 'r', encoding='utf-8') as f:
    dataset = f.read()

sent_tokens = nltk.sent_tokenize(dataset)
word_tokens = nltk.word_tokenize(dataset)
lemmatizer = nltk.stem.WordNetLemmatizer()

def preprocess(tokens):
    return [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]

corpus = [" ".join(preprocess(nltk.word_tokenize(sentence))) for sentence in sent_tokens]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

def chatbot_response(user_input):
    # Preprocess user input
    user_input = " ".join(preprocess(nltk.word_tokenize(user_input)))

    # Vectorize user input
    user_vector = vectorizer.transform([user_input])

    # Calculate cosine similarity between user input and corpus
    similarities = cosine_similarity(user_vector, X)

    # Get index of most similar sentence
    idx = similarities.argmax()

    # Return corresponding sentence from corpus
    return sent_tokens[idx]

# Create a Streamlit app with the updated chatbot function
def main():
    st.title("Butt, the Bot")
    st.write("Hello there! My name is Butt, and I'm a simple Bot!\n\n You can ask me some simple data science or data science related terminologies, nothing more! \n\n You can also check out my chat history with other visitors. Cheers, my gee!")
    # Get the user's question
    question = st.text_input("You:")
    with open('chat_history.txt', 'a') as f:
        message = question
        timestamp = datetime.datetime.now()
        f.write(f"{timestamp} User: {message}\n")
    # Create a button to submit the question
    if st.button("Submit"):
        with st.spinner('Generating response...'):
            time.sleep(2)
        response = chatbot_response(question)
        with open('chat_history.txt', 'a') as f:
            message = response
            timestamp = datetime.datetime.now()
            f.write(f"{timestamp} Butt: {message}\n")
        st.write("Chatbot: " + response)

    with open('chat_history.txt') as file:
        lines = file.readlines()

    chat_history = []
    for line in lines:
        line = line.strip()
        if ':' in line:
            timestamp, message = line.split(':', 1)
            chat_history.append({'timestamp': timestamp, 'message': message})
        else:
            # handle the case where there are not exactly 2 values in the line
            pass

    df = pd.DataFrame(chat_history)
    st.write(df)

if __name__ == "__main__":
    main()