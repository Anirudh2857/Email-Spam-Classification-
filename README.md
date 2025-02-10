# Email Spam Classification

## Overview
This project focuses on classifying emails as spam or ham (not spam) using various machine learning techniques. The dataset used in this project contains email texts labeled as spam (1) or ham (0).

## Dataset
- The dataset is loaded from `emails.csv`
- It contains text messages along with a spam label (1 for spam, 0 for ham)

## Preprocessing Steps
1. **Exploratory Data Analysis (EDA)**
   - Data is explored using `pandas` and `seaborn`
   - Message lengths are analyzed
   - Spam and ham distributions are plotted

2. **Text Cleaning**
   - Removing punctuation
   - Removing stopwords using NLTK

3. **Feature Extraction**
   - Applying `CountVectorizer` to convert text data into a matrix of token counts
   - Applying `TfidfTransformer` to transform the count matrix into a TF-IDF representation

## Model Training
### Naive Bayes Classifier
- Trained using `MultinomialNB` on the transformed text data
- Performance evaluated using confusion matrix and classification report

### Alternative Models Used
- **Decision Tree Classifier**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**

Each model is trained on the processed dataset and evaluated using performance metrics such as precision, recall, and F1-score.

## Results & Evaluation
- Confusion matrices and classification reports are generated for each model
- Naive Bayes and SVM perform well due to their efficiency in text classification
- Results are visualized using `seaborn` heatmaps

## Dependencies
Ensure the following libraries are installed before running the notebook:
```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn
```

## Running the Notebook
To execute the notebook, open it in Google Colab or Jupyter Notebook and run each cell sequentially.

## Conclusion
This project demonstrates various machine learning approaches for email spam classification. The Naive Bayes model performs particularly well due to its suitability for text classification tasks.

