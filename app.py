import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import word_tokenize, sent_tokenize, pos_tag, FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer

# Download resources (only first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('omw-1.4')

# Title
st.title("🧠 NLP Text Analyzer App")

# Input text
text = st.text_area("Enter your text here:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # 1. Sentence Tokenization
        sentences = sent_tokenize(text)
        st.subheader("📌 Sentences")
        st.write(sentences)

        # 2. Word Tokenization
        words = word_tokenize(text)
        st.subheader("🔤 Words")
        st.write(words)

        # 3. Stopword Removal
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in words if w.lower() not in stop_words and w.isalpha()]
        st.subheader("🚫 Stopwords Removed")
        st.write(filtered_words)

        # 4. Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(w.lower()) for w in filtered_words]
        st.subheader("🧾 Lemmatized Words")
        st.write(lemmatized)

        # 5. Stemming
        stemmer = PorterStemmer()
        stemmed = [stemmer.stem(w.lower()) for w in filtered_words]
        st.subheader("🌱 Stemmed Words")
        st.write(stemmed)

        # 6. POS Tagging
        pos_tags = pos_tag(words)
        st.subheader("🏷 POS Tags")
        st.write(pos_tags)

        # 7. Frequency Distribution
        freq_dist = FreqDist(lemmatized)
        st.subheader("📊 Word Frequency")
        st.write(freq_dist.most_common())

        # 8. Sentiment Analysis
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)

        st.subheader("😊 Sentiment Analysis")
        st.write(sentiment)

        if sentiment['compound'] >= 0.05:
            st.success("Positive 😊")
        elif sentiment['compound'] <= -0.05:
            st.error("Negative 😡")
        else:
            st.info("Neutral 😐")