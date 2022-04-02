import numpy as np
import re, string
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ast
from wordcloud import WordCloud, STOPWORDS
import random
from annotated_text import annotated_text
import spacy
from datasets import load_dataset, load_from_disk

# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("omw-1.4")
# nltk.download("wordnet")
nlp = spacy.load("en_core_web_sm")


@st.cache(allow_output_mutation=True)
def sentimentAnalysisModel():
    return pipeline("sentiment-analysis", model="bertweet_model")


@st.cache(allow_output_mutation=True)
def machineTranslationModel():
    return pipeline("translation", model="translator_model")


def visWordFreq(df, col):

    st.subheader("Word Frequency")

    all_words = []
    for i in range(len(df)):
        all_words += df[col][i]

    nlp_words = nltk.FreqDist(all_words)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = nlp_words.plot(20, color="salmon")
    with st.expander("Show graph"):
        st.pyplot(fig)

    st.subheader("Bigram Frequency")
    bigram = nltk.bigrams(all_words)
    bigram_words = nltk.FreqDist(bigram)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = bigram_words.plot(20, color="green")
    with st.expander("Show graph"):
        st.pyplot(fig)

    st.subheader("Trigram Frequency")
    trigram = nltk.trigrams(all_words)
    trigram_words = nltk.FreqDist(trigram)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = trigram_words.plot(20, color="blue")
    with st.expander("Show graph"):
        st.pyplot(fig)

    st.subheader("Word Cloud")
    word = " ".join(all_words) + " "
    wordcloud = WordCloud().generate(word)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = plt.imshow(wordcloud)
    plt.axis("off")
    with st.expander("Show Word Cloud"):
        st.pyplot(fig)


@st.cache
def cleanText(text):
    sw = stopwords.words("english")
    text = str(text)
    text = text.lower()
    text = re.sub("@", "", text)
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    text = re.sub(r"[^a-zA-Z ]+", "", text)
    text = nltk.word_tokenize(text)
    text = [w for w in text if w not in sw]
    return text


@st.cache
def lemma(text):
    lemma = WordNetLemmatizer()
    text = [lemma.lemmatize(t) for t in text]
    text = [lemma.lemmatize(t, "v") for t in text]
    return text


@st.cache(ttl=24 * 60 * 60, allow_output_mutation=True)
def returnDF(df, col):
    newdf = pd.DataFrame(df[col], columns=[str(col)])
    newdf[col] = newdf[col].apply(cleanText)
    newdf[col] = newdf[col].apply(lambda x: lemma(x))
    return newdf


def generateRandom(ldf):
    yield (random.randint(0, ldf))


# @st.cache(suppress_st_warning=True)
def dfSentimentAnalysis(df, col):
    st.subheader("Dataset Sentiments")
    st.caption("10 random rows were chosen from the dataset")
    bt = sentimentAnalysisModel()
    df = df[df[col].map(lambda x: len(x)) > 0]
    df = df.reset_index(drop=True)
    ndf = pd.DataFrame(df[col], columns=[str(col), "sentiment"])
    ndf = ndf.reset_index(drop=True)
    df = df.reset_index(drop=True)
    # st.dataframe(ndf)
    resdf = pd.DataFrame(columns=[str(col), "sentiment"])
    for i in range(10):
        row = random.randint(0, len(ndf))
        res = bt(ndf[col][row])
        ndf.loc[row, "sentiment"] = res[0]["label"]
        resdf.loc[-1] = ndf.iloc[row]
        resdf.index = resdf.index + 1
        resdf = resdf.reset_index(drop=True)
    st.dataframe(resdf)


def translate(res):
    for i in range(5):
        res.loc[-1] = mt_model(df[col][i])[0]["translation_text"]
        res.index += 1
        res = res.sort_index()
    return res


def absa(df, col):
    st.subheader("Annotated Text")
    st.write("5 random rows were chosen from the dataset")
    st.write("The sentences are annotated as such,")
    annotated_text("This is a ", ("positive ", "", "#117d00"), "sentence.")
    annotated_text("This is a ", ("negative ", "", "#ff0019"), "sentence.")
    annotated_text("This is a ", ("neutral ", "", "#ffffff", "#000000"), "sentence.")
    st.write("")
    numRows = []
    sentences = []
    for i in range(5):
        numRows.append(random.randint(0, len(df)))

    for i in numRows:
        sentences.append(df[col][i][:300])

    aspect = []
    for sentence in sentences:
        doc = nlp(sentence)
        descriptive_term = ""
        target = ""
        for token in doc:
            if token.dep_ == "nsubj" and token.pos_ == "NOUN":
                target = token.text
            if token.pos_ == "ADJ":
                prepend = ""
                for child in token.children:
                    if child.pos_ != "ADV":
                        continue
                    prepend += child.text + " "
                descriptive_term = prepend + token.text
        aspect.append(descriptive_term)

    bt = sentimentAnalysisModel()
    with st.expander("Show annotated text"):
        for i in range(len(sentences)):
            sa = bt(sentences[i])[0]["label"]
            colourString = ""
            textColour = "#ffffff"
            if sa == "POS":
                colourString = "#117d00"
            elif sa == "NEG":
                colourString = "#ff0019"
            else:
                colourString = "#ffffff"
                textColour = "#000000"

            idx = sentences[i].index(aspect[i])
            annotated_text(
                sentences[i][:idx],
                (
                    sentences[i][idx : idx + len(aspect[i])],
                    "",
                    colourString,
                    textColour,
                ),
                sentences[i][idx + len(aspect[i]) :],
                " ...",
            )
            st.write("")


def absaTranslated(df, col):
    st.subheader("Annotated Text")
    st.write("5 random rows were chosen from the dataset")
    st.write("The sentences are annotated as such,")
    annotated_text("This is a ", ("positive ", "", "#117d00"), "sentence.")
    annotated_text("This is a ", ("negative ", "", "#ff0019"), "sentence.")
    annotated_text("This is a ", ("neutral ", "", "#ffffff", "#000000"), "sentence.")
    st.write("")
    sentences = []

    for i in range(5):
        sentences.append(df[col][i][:300])

    aspect = []
    for sentence in sentences:
        doc = nlp(sentence)
        descriptive_term = ""
        target = ""
        for token in doc:
            if token.dep_ == "nsubj" and token.pos_ == "NOUN":
                target = token.text
            if token.pos_ == "ADJ":
                prepend = ""
                for child in token.children:
                    if child.pos_ != "ADV":
                        continue
                    prepend += child.text + " "
                descriptive_term = prepend + token.text
        aspect.append(descriptive_term)

    bt = sentimentAnalysisModel()
    with st.expander("Show annotated text"):
        for i in range(len(sentences)):
            sa = bt(sentences[i])[0]["label"]
            colourString = ""
            textColour = "#ffffff"
            if sa == "POS":
                colourString = "#117d00"
            elif sa == "NEG":
                colourString = "#ff0019"
            else:
                colourString = "#ffffff"
                textColour = "#000000"

            idx = sentences[i].index(aspect[i])
            annotated_text(
                sentences[i][:idx],
                (
                    sentences[i][idx : idx + len(aspect[i])],
                    "",
                    colourString,
                    textColour,
                ),
                sentences[i][idx + len(aspect[i]) :],
                " ...",
            )
            st.write("")


with st.sidebar:
    choose = option_menu(
        "Natural Language Processing",
        [
            "Sentence Sentiment Analysis",
            "Dataset Analysis",
            "Dataset Analysis (Non-English)",
        ],
        menu_icon="cpu-fill",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#004ead"},
            "icon": {"color": "#ea00ff", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#babfbe",
            },
            "nav-link-selected": {"background-color": "#00185c"},
        },
    )

if choose == "Sentence Sentiment Analysis":
    st.title("Sentence Sentiment Analysis")
    sent = st.text_input("Enter text here: ")
    bt = sentimentAnalysisModel()
    res = bt(sent)
    if res[0]["label"] == "NEG":
        st.write("Negative, Score: " + str(res[0]["score"]))
    elif res[0]["label"] == "POS":
        st.write("Positive, Score: " + str(res[0]["score"]))
    else:
        st.write(res[0]["label"])

elif choose == "Dataset Analysis":
    st.title("Dataset Analysis")
    option = st.selectbox(
        "Select a dataset or upload your own", ["Amazon", "IMDB", "Upload"]
    )

    if option == "Upload":
        uploaded_file = st.file_uploader("Upload your own dataset")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader(f"Loaded Dataset - {uploaded_file.name} - {len(df)} rows")
            st.dataframe(df.head())
            if len(df) > 1000:
                st.write(
                    "Dataset too large to process, processing only the top 1000 rows..."
                )
                df = df.head(1000)
            col = st.selectbox("Select a column", df.columns)
            if st.button("Process"):
                newdf = returnDF(df, col)
                st.dataframe(newdf.head())
                visWordFreq(newdf, col)
                dfSentimentAnalysis(newdf, col)
                absa(df, col)
    elif option == "Amazon":
        df = pd.read_csv("amazon.csv")
        st.subheader(f"Loaded Dataset - Amazon - {len(df)} rows")
        st.dataframe(df.head())
        if len(df) > 1000:
            st.write(
                "Dataset too large to process, processing only the top 1000 rows..."
            )
            df = df.head(1000)
        col = st.selectbox("Select a column", df.columns)
        if st.button("Process"):
            newdf = returnDF(df, col)
            st.dataframe(newdf.head())
            visWordFreq(newdf, col)
            dfSentimentAnalysis(newdf, col)
            absa(df, col)
    elif option == "IMDB":
        df = pd.read_csv("imdb_review.csv")
        st.subheader(f"Loaded Dataset - IMDB - {len(df)} rows")
        st.dataframe(df.head())
        if len(df) > 1000:
            st.write(
                "Dataset too large to process, processing only the top 1000 rows..."
            )
            df = df.head(1000)
        col = st.selectbox("Select a column", df.columns)
        if st.button("Process"):
            newdf = returnDF(df, col)
            st.dataframe(newdf.head())
            visWordFreq(newdf, col)
            dfSentimentAnalysis(newdf, col)
            absa(df, col)


else:
    st.title("Dataset Analysis (Non-English)")
    uploaded_file = st.file_uploader("Upload your own dataset")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.dropna(inplace=True)
        st.subheader(f"Loaded Dataset - {uploaded_file.name} - {len(df)} rows")
        st.dataframe(df.head())
        if len(df) > 1000:
            st.write(
                "Dataset too large to process, processing only the top 1000 rows..."
            )
            df = df.head(1000)

        col = st.selectbox("Select a column", df.columns)
        if st.button("Process"):
            mt_model = machineTranslationModel()
            st.subheader("Dataframe with chosen column")
            st.dataframe(df[col])

            res = pd.DataFrame(columns=["trans_review"])
            with st.spinner("Loading..."):
                res = translate(res)
                resN = returnDF(res, "trans_review")
                st.subheader("Translated Review")
                st.dataframe(resN)
                visWordFreq(resN, "trans_review")
                st.dataframe(res)
                # dfSentimentAnalysis(res, "trans_review")
                absaTranslated(res, "trans_review")
