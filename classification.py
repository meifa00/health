import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity

# Replace with your OpenAI API key
openai.api_key = st.secrets["mykey"]

def load_data():
    df = pd.read_csv("qa_dataset_with_embeddings.csv")
    questions = df['Question'].tolist()
    embeddings = np.array(df['Question_Embedding'].tolist())
    return df, questions, embeddings

df, questions, embeddings = load_data()

def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

def find_best_match(user_question, questions, embeddings, threshold=0.7):
    query_embedding = get_embedding(user_question)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    best_match_index = np.argmax(similarities)
    best_match_similarity = similarities[best_match_index]

    if best_match_similarity >= threshold:
        return df.iloc[best_match_index]['Answer'], best_match_similarity
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", None
        
def main():
    st.title("Heart, Lung, and Blood Health Assistant")

    user_question = st.text_input("Ask your question here:")
    if st.button("Submit"):
        answer, similarity = find_best_match(user_question, questions, embeddings)
        st.success(answer)
        if similarity:
            st.write(f"Similarity Score: {similarity:.2f}")

if __name__ == "__main__":
    main()
