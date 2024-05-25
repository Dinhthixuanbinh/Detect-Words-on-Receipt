# app.py
import streamlit as st
from inference import ModelInference

@st.cache
def load_model(model_path):
    return ModelInference(model_path)

def main():
    st.title("Sentiment Analysis App")
    st.write("Enter a text to analyze its sentiment:")

    text = st.text_input("Text")

    if st.button("Analyze"):
        model = load_model('/content/best_model.pth')
        sentiment = model.predict(text)
        st.write("Sentiment:", sentiment)

if __name__ == "__main__":
    main()
