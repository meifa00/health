import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss  # For efficient nearest neighbor search

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error("Dataset file not found.")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("Dataset is empty.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

def load_embeddings(data):
    embeddings = np.array(data['Question_Embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).tolist())
    return embeddings

def initialize_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Consider other models
    return model

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def find_similar_questions(user_embedding, index, embeddings, data, threshold=0.7):
    distances, indices = index.search(user_embedding, k=1)
    most_similar_idx = indices[0][0]
    most_similar_score = 1 - distances[0][0]  # Convert distance to similarity

    if most_similar_score > threshold:
        answer = data.iloc[most_similar_idx]['Answer']
        return answer, most_similar_score
    else:
        return None, None

def main():
    st.title("Health Q&A System")

    # Load data and embeddings
    data = load_data('qa_dataset_with_embeddings.csv')
    embeddings = load_embeddings(data)

    # Initialize embedding model
    model = initialize_model()

    # Create Faiss index
    faiss_index = build_faiss_index(embeddings)

    # User input for question
    user_question = st.text_input("Ask a question about heart, lung, or blood-related health:")

    # Button to search for answers
    if st.button("Get Answer"):
        if user_question:
            user_embedding = model.encode([user_question])
            user_embedding = np.array(user_embedding)

            answer, similarity_score = find_similar_questions(user_embedding, faiss_index, embeddings, data)

            if answer:
                st.subheader("Answer:")
                st.write(answer)
                st.write(f"Similarity Score: {similarity_score:.2f}")
            else:
                st.write("I apologize, but I don't have information on that topic yet. Could you please ask other questions?")
        else:
            st.write("Please enter a question.")

    # Clear input field button
    if st.button("Clear"):
        st.text_input("Ask a question about heart, lung, or blood-related health:", value="", key="clear")

if __name__ == "__main__":
    main()
