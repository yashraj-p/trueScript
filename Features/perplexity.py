##   perplexity


import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import numpy as np
import re
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')

sample_df = pd.read_csv('/content/Hc3Finalfeatures.csv')

label_counts = sample_df['label'].value_counts()

print(label_counts)

sample_df = sample_df[['text', 'label']]

sample_df['text'] = sample_df['text'].apply(lambda x: ' '.join(eval(x)))

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Removing brackets and special characters
    text = re.sub(r'\[.*?\]', '', text)
    # Removing specific special characters: < > \ / , '
    text = re.sub(r'[<>\\/,\'"]', '', text)
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenizing
    tokens = word_tokenize(text)
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

sample_df['text'] = sample_df['text'].apply(preprocess_text)

sample_df.head()


sample_df['sentences'] = sample_df['text'].apply(sent_tokenize)

sample_df.head()


all_sentences = [word_tokenize(sentence) for sentences in sample_df['sentences'] for sentence in sentences]


bigrams = [(w1, w2) for sentence in all_sentences for w1, w2 in zip(sentence[:-1], sentence[1:])]


bigram_freq = Counter(bigrams)
unigram_freq = Counter([word for sentence in all_sentences for word in sentence])

def calculate_perplexity(sentence):
    tokens = word_tokenize(sentence)
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    vocab_size = len(unigram_freq)

    probs = []
    for w1, w2 in bigrams:
        prob = (bigram_freq[(w1, w2)] + 1) / (unigram_freq[w1] + vocab_size)  # Add-1 smoothing
        probs.append(prob)

    if probs:
        perplexity = np.exp(-np.mean(np.log(probs)))
        return perplexity
    else:
        return np.nan

def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    return pd.Series(sentences, index=[f'sentence{i+1}' for i in range(len(sentences))])

tokenized_df = sample_df['text'].apply(tokenize_sentences)
final_df = pd.concat([tokenized_df, sample_df[['label']]], axis=1)

def apply_perplexity(row):
    sentence_cols = [col for col in row.index if col.startswith('sentence')]
    perplexities = [calculate_perplexity(row[col]) for col in sentence_cols if not pd.isna(row[col])]
    return pd.Series({
        'mean_perplexity': np.mean(perplexities),
        'median_perplexity': np.median(perplexities),
        'std_perplexity': np.std(perplexities)
    })

perplexity_results = final_df.apply(apply_perplexity, axis=1)

perplexity_results_df = pd.DataFrame(perplexity_results, columns=['mean_perplexity', 'median_perplexity', 'std_perplexity'])

final_df = pd.concat([sample_df['text'], perplexity_results_df], axis=1)

final_df.head(500)

from google.colab import files
final_df.to_csv('final_df.csv', index=True)


files.download('final_df.csv')


# output_path = '/content/diff_perplexity_df_final.csv'
# result_df.to_csv(output_path, index=False)



