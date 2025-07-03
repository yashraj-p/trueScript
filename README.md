# trueScript
AI Text Detection
# trueScript - Distinguishing Between AI-Generated and Human-Written Texts

## Project Overview

The primary objective of this project is to develop a robust machine learning model capable of distinguishing between **AI-generated text** and **human-written text**. The models are trained on a diverse dataset containing examples of both types of text, and it will employ state-of-the-art natural language processing (NLP) techniques to achieve high accuracy.

## Key Features

- **Data Preparation**: Diverse HC3 dataset containing both AI-generated and human-written texts has been used
- **Natural Language Processing**: Utilized advanced NLP techniques such as tokenization, embedding, and feature extraction.
- **Pre-trained Models**: Leverages models like BERT, RoBERTa, GPT-2 and CNN-BiLSTM architecture for fine-tuning on classification tasks.
- **Additional Features**: Incorporated text readability metrics and linguistic features to boost model performance.
- **Performance Evaluation**: Evaluated the model using metrics like accuracy, precision, recall, and F1 score.

## Project Workflow

1. **Data Collection**: Hc3 Dataset has been used. 90-10 data split has been done. 90% for fine tuning and 10% for testing of fine tuned model on unseen data.
2. **Feature Engineering**: Extracted features:
   - Average line length
   - Word density
   - Mean perplexity
   - Readability scores (Flesch-Kincaid and Gunning Fog)
   - Burstiness
3. **Model Development**:
   - Fine-tune pre-trained models (BERT, RoBERTa, GPT-2 and CNN-BiLSTM architecture).
   - Train the model using the engineered features.
4. **Evaluation and Testing**:
   - Test the model on unseen data (10% testing data).
   - Evaluate performance using classification metrics.

## Conclusion

**After testing on unseen 10% data, BERT Model performed best with accuracy of 97.13%**
