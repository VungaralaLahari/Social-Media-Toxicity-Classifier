# Social Media Toxicity Classifier

Content moderation at scale is a real problem. Platforms like YouTube and Reddit receive millions of comments daily - manual review doesn't scale. This project builds a two-stage ML pipeline that detects whether a comment is toxic, and if yes, identifies what kind.

 ---
 
## What it does

Stage 1 classifies a comment as toxic or clean.

Stage 2 runs only if toxic is detected, and identifies the type: threat, insult, obscene, identity hate, or severe toxic.

Built a Streamlit app where you paste any comment and instantly get the verdict, confidence score, and category breakdown.

**Live Demo:** [https://social-media-toxicity-classifier-7a3q6zdlxmx8yqbjjlj3db.streamlit.app/](https://social-media-toxicity-classifier-7a3q6zdlxmx8yqbjjlj3db.streamlit.app/)

---

## Dataset

Jigsaw Toxic Comment Classification (Kaggle) - 159,571 Wikipedia comments with 6 toxicity labels. Multi-label, meaning one comment can belong to multiple categories simultaneously.

Key stat: 89.8% of comments are clean, 10.2% toxic. Severe class imbalance from the start.

---

## Why two stages?

A single multi-label model trained on all 159K comments struggles because the overwhelming majority are clean. Separating binary detection from type classification keeps each model focused on what it needs to learn.

---

## Key decisions

### Threshold set to 0.4 not 0.5

Default threshold of 0.5 gave recall of 0.85 on toxic class. Lowering to 0.4 pushed recall to 0.88. In content moderation, missing a toxic comment causes user harm. A few extra false flags are an acceptable tradeoff.

### Preserved pronouns and negations in stopword removal

Standard NLTK stopwords remove "you", "not", "will", "never". But "you will die" loses its entire meaning if those words are stripped. Kept them intentionally.

### TF-IDF with bigrams

ngram_range=(1,2) captures two-word phrases like "kill you" as a single feature. More toxic signal than treating each word independently.

### SMOTE only on Stage 1

Stage 2 uses class_weight='balanced' instead. Applying SMOTE to multi-label targets is complex and class_weight handles within-toxic imbalance well enough.

### F1 and ROC-AUC, not accuracy

With 89.8% clean data, a model predicting "clean" for everything scores 89.8% accuracy while catching zero toxic comments. Accuracy is meaningless here.

---

## Model comparison (Stage 1)

| Model               | F1 Score | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | 0.6555   | 0.9490  |
| Random Forest       | 0.6846   | 0.9402  |

Went with Logistic Regression. ROC-AUC is higher (0.949 vs 0.940) and it runs significantly faster on sparse TF-IDF matrices. F1 is slightly lower but the AUC advantage matters more for this use case.

---

## Stage 1 results (threshold = 0.4)

| Class | Precision | Recall | F1   |
| ----- | --------- | ------ | ---- |
| Clean | 0.98      | 0.89   | 0.93 |
| Toxic | 0.46      | 0.88   | 0.61 |

## Stage 2 results (multi-label)

| Label         | Precision | Recall | F1   |
| ------------- | --------- | ------ | ---- |
| toxic         | 0.61      | 0.87   | 0.71 |
| obscene       | 0.63      | 0.88   | 0.74 |
| insult        | 0.52      | 0.89   | 0.65 |
| severe_toxic  | 0.25      | 0.85   | 0.38 |
| identity_hate | 0.20      | 0.77   | 0.31 |
| threat        | 0.16      | 0.77   | 0.27 |

Threat and identity_hate have weak precision because they have very few training samples (74 and 294 respectively out of 31K test samples). The model learns to cast a wide net, which hurts precision but keeps recall high.

## What failed

Initial model without SMOTE had 90% accuracy but near-zero recall on the toxic class. It was predicting "clean" for almost everything and getting rewarded for it. SMOTE + threshold tuning fixed this.

Threat category still has poor precision (0.16). With only 474 training samples, the model does not have enough signal to be confident. More data or domain-specific embeddings would help here.

---

## Stack

Python, Pandas, NumPy, NLTK, Scikit-learn, Imbalanced-learn, Streamlit, Joblib

## Project structure

```bash
toxicity-classifier/
├── Social_Media_Toxicity_Classifier_Notebook.ipynb
├── app.py
├── tfidf.pkl
├── binary_model.pkl
├── multilabel_model.pkl
├── requirements.txt
└── README.md
```

## Run it

```bash
git clone https://github.com/VungaralaLahari/toxicity-classifier.git
cd toxicity-classifier
pip install -r requirements.txt
streamlit run app.py
```
---

## What I would improve next

Replace TF-IDF with DistilBERT embeddings. TF-IDF treats "I love this" and "I love hurting this" almost identically. Context matters for toxicity and transformers handle it better.

Collect more samples for threat and identity_hate categories. These labels have the weakest performance purely because of data scarcity.

---

**Vungarala Lahari**
