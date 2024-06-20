# sentiment_analysis_app.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained("assemblyai/distilbert-base-uncased-sst2")
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized_text.input_ids
    attention_mask = tokenized_text.attention_mask

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=1).squeeze()
    return probs[1].item(), probs[0].item()  # Positive probability, Negative probability

def main():
    st.title("Sentiment Analysis with Transformers")
    st.write("Enter some text and get the sentiment prediction.")

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()

    # Input text box
    input_text = st.text_area("Input Text", "Type your text here...")

    if st.button("Predict Sentiment"):
        if input_text:
            positive_prob, negative_prob = predict_sentiment(input_text, tokenizer, model)
            st.write(f"Positive probability: {positive_prob:.2f}")
            st.write(f"Negative probability: {negative_prob:.2f}")
        else:
            st.write("Please enter some text to get a prediction.")

if __name__ == "__main__":
    main()
