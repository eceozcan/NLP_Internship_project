# NLP_Internship_project
SafeSpeech Software Intern project

# Sentiment Analysis on Winvoker Reviews & Airline Tweets  

## 📌 Project Overview  
This project focuses on **sentiment analysis (duygu analizi)** using two different datasets:  

1. **Winvoker Reviews (Türkçe)**  
2. **Airline Tweets (English)**  

The workflow includes:  
- Data collection & loading  
- Text preprocessing (Zemberek for Turkish, NLTK for English)  
- Vectorization using **BERT embeddings**  
- Model training with different ML algorithms  
- Performance evaluation and visualization  


## 📂 Datasets  

### 🇹🇷 Winvoker Dataset (Turkish)  
- Loaded from HuggingFace datasets.  
- Split into `train_df` and `test_df`.  
- Columns:  
  - `text`: User review  
  - `label`: Sentiment label (positive, negative, neutral)  

**Preprocessing**:  
- Lowercasing  
- Removing punctuation and special characters  
- Tokenization  
- Lemmatization using **Zemberek**  
- Stop-word removal (optional)  
- Processed data stored with **joblib**  

**Vectorization**:  
- Model: `dbmdz/bert-base-turkish-cased`  
- Each text → 768-dimensional embeddings  

**Models trained**:  
- Logistic Regression (LR)  
- Support Vector Machine (SVM)  
- Multi-Layer Perceptron (MLP)  


### 🇬🇧 Airline Tweets Dataset (English)  
- Columns:  
  - `text`: Tweet text  
  - `airline_sentiment`: Sentiment label (positive, negative, neutral)  

**Preprocessing with NLTK**:  
- Lowercasing  
- Removing URLs, mentions, hashtags  
- Removing punctuation and numbers  
- Tokenization  
- Stop-word removal  
- Lemmatization  

**Vectorization**:  
- BERT embeddings  

**Models trained**:  
- Logistic Regression (LR)  
- Support Vector Machine (SVM)  
- Multi-Layer Perceptron (MLP)  
- Random Forest (RF)  


## 📊 Results  

### Winvoker Dataset  
- **Logistic Regression**: 93.44%  
- **SVM**: 93.31%  
- **MLP**: 94.21% (best overall)

## 📈 Visualizations  

### Winvoker Dataset  
![Winvoker Graph](Çalışma%201/winvoker-airlines_ekran%20görüntüleri/winvoker_graph.png)  

### Airline Tweets Dataset  
![Airline Graph](Çalışma%201/winvoker-airlines_ekran%20görüntüleri/Airline_model_v1_graph_all.jpg)  


🔎 Notes:  
- MLP achieved the best accuracy and F1 on the positive class.  
- LR & SVM had lower recall in the negative class (~66–68%).  
- Convergence warnings observed → could improve with more iterations or scaling.  


### Airline Tweets Dataset  
- **Logistic Regression**: 76.91%  
- **SVM**: 77.22% (best overall)  
- **MLP**: 74.01% (after increasing iterations; previously 72%)  
- **Random Forest**: 74.25%  

🔎 Notes:  
- LR & SVM had balanced performance but **low recall for neutral class (~52–53%)**.  
- MLP & RF performed well on negative class, but weaker on positive/neutral.  
- RF was weaker on macro F1 due to class imbalance.  


## 📈 Key Insights  
- **Winvoker dataset**: Balanced and larger → higher accuracy (best model: **MLP**).  
- **Airline dataset**: Imbalanced classes → performance varied by sentiment.  
- **BERT embeddings**: Provided strong text representations, outperforming classical TF-IDF approaches.  
- Visualizations clearly showed model comparisons and training curves.  


## 🚀 Technologies Used  
- **Python**  
- **HuggingFace Datasets & Transformers**  
- **Zemberek NLP (for Turkish lemmatization)**  
- **NLTK (for English preprocessing)**  
- **scikit-learn (ML models: LR, SVM, MLP, RF)**  
- **joblib (data persistence)**  
- **Matplotlib / VS Code** (visualizations)  


## 📜 Conclusion  
This project demonstrates how sentiment analysis pipelines differ between **Turkish** and **English** datasets.  
- Preprocessing steps need to be adapted per language.  
- BERT embeddings significantly improve model performance.  
- Dataset balance has a strong effect on accuracy and F1-scores.  
- Among all experiments, **MLP on Winvoker dataset** achieved the highest performance.  
