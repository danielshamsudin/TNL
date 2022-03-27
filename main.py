import streamlit as st
import pandas as pd
from transformers import pipeline

bt = pipeline("sentiment-analysis", model="bertweet_model")
st.title("Sentiment Analysis")

sent = st.text_input("Enter text here: ")
res = bt(sent)

if res[0]["label"] == "NEG":
    st.write("Negative, Score: " + str(res[0]["score"]))
else:
    st.write("Positive, Score: " + str(res[0]["score"]))
