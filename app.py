import streamlit as st
import joblib
import re
import emoji

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load saved models
tfidf        = joblib.load("tfidf.pkl")
binary_model = joblib.load("binary_model.pkl")
multi_model  = joblib.load("multilabel_model.pkl")

# Threshold tuned for higher recall (content moderation — missing toxic is worse)
THRESHOLD = 0.4

# Stopwords — preserve words critical to toxicity meaning
stop_words = set(stopwords.words('english'))
important_words = {
    'you', 'your', 'yours', 'yourself',
    'he', 'she', 'they', 'them', 'him', 'her',
    'not', 'no', 'never',
    'will', 'shall', 'must', 'should'
}
stop_words = stop_words - important_words

l = WordNetLemmatizer()

def text_preprocess(text):
    text = text.strip()
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)
    text = emoji.replace_emoji(text, replace='')
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [l.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

label_descriptions = {
    'toxic'        : 'General Toxic',
    'severe_toxic' : 'Severely Toxic',
    'obscene'      : 'Obscene Language',
    'threat'       : 'Threat / Violence',
    'insult'       : 'Personal Insult',
    'identity_hate': 'Identity-Based Hate'
}

# UI
st.set_page_config(page_title="Toxicity Classifier")
st.title(" Social Media Toxicity Classifier")
st.markdown("Detects toxic content and classifies the type of toxicity.")

user_input = st.text_area("Enter a comment to analyze:", height=120,
                           placeholder="Type or paste a comment here...")

if st.button("Analyze", type="primary"):

    if not user_input or not user_input.strip():
        st.warning("Please enter a comment before analyzing.")

    elif len(user_input.strip()) < 5:
        st.warning("Comment is too short to analyze meaningfully.")

    else:
        with st.spinner("Analyzing..."):

            clean_input  = text_preprocess(user_input)
            vector_input = tfidf.transform([clean_input])

            # Stage 1 — binary
            confidence   = binary_model.predict_proba(vector_input)[0][1]
            binary_pred  = int(confidence > THRESHOLD)

        if binary_pred == 0:
            st.success(" Non-Toxic Comment")
            st.metric("Toxicity Confidence", f"{confidence:.1%}")
            st.caption(f"Below threshold of {THRESHOLD} — classified as clean.")

        else:
            st.error(" Toxic Comment Detected")
            st.metric("Toxicity Confidence", f"{confidence:.1%}")

            # Stage 2 — multi-label
            multi_pred  = multi_model.predict(vector_input)[0]
            toxic_types = [label_descriptions[label_cols[i]]
                           for i, val in enumerate(multi_pred) if val == 1]

            if toxic_types:
                st.markdown("**Toxicity Categories Detected:**")
                for t in toxic_types:
                    st.markdown(f"- {t}")
            else:
                st.markdown("**Category:** General toxic content")

st.markdown("---")
st.caption("Model: Logistic Regression + TF-IDF | Threshold: 0.4 (tuned for recall) | Dataset: Jigsaw Toxic Comment Classification")