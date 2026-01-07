# Turkish Sentiment Analysis on Streaming App Reviews

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20TensorFlow-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project aims to analyze user sentiment (Positive/Negative) from Turkish reviews of major digital streaming platforms (Netflix, Disney+, Amazon Prime, Spotify, Twitch). By employing a "Data Science Pipeline," the project covers **data collection (web scraping)**, **advanced preprocessing (stemming)**, and **comparative modeling** using both Traditional Machine Learning and Deep Learning architectures.

## 📂 Dataset & Data Collection
The dataset was built from scratch using the `google-play-scraper` library.
* **Source:** Google Play Store
* **Target Apps:** Netflix, Disney+, Amazon Prime Video, Spotify, Twitch
* **Size:** ~2,800 Reviews
* **Labeling:**
    * ⭐️ 1-2 Stars → `Negative` (0)
    * ⭐️ 4-5 Stars → `Positive` (1)
    * (Neutral reviews were excluded for binary classification)

> **⚠️ Ethical Considerations & Terms of Service:**
> The data collection process was conducted using the `google-play-scraper` library strictly for academic research purposes. No personal identifiable information (PII) of the users was stored or processed. The scraping rate was limited to respect the server integrity of the Google Play Store, complying with fair use policies for non-commercial sentiment analysis.

## 🛠️ Technologies & Tools
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **NLP & Preprocessing:** NLTK, **TurkishStemmer** (for agglutinative morphology), Re (Regex)
* **Machine Learning:** Scikit-learn (Logistic Regression, Random Forest, SVM)
* **Deep Learning:** TensorFlow/Keras (Sequential MLP)
* **Visualization:** Matplotlib, Seaborn

## ⚙️ Methodology & Pipeline

### 1. Preprocessing
Since Turkish is an agglutinative language, a robust cleaning pipeline was implemented:
* **Noise Removal:** Regex to remove punctuation and numbers.
* **Stopwords:** Removing common Turkish stopwords via NLTK.
* **Stemming:** Using `TurkishStemmer` to reduce words to their roots (e.g., *geliyorum* → *gel*), significantly reducing feature sparsity.

### 2. Feature Extraction
* **TF-IDF (Term Frequency-Inverse Document Frequency):** Used to convert text data into numerical vectors (Max features: 2,000).

### 3. Model Architecture
Four different models were trained and compared:
* **Logistic Regression:** Baseline linear model (GridSearch used for Regularization tuning).
* **Random Forest:** Ensemble method to reduce variance.
* **SVM (Support Vector Machine):** Linear kernel for high-dimensional text data.
* **Deep Learning (MLP):** A custom Keras Neural Network with:
    * 2 Hidden Layers (ReLU activation)
    * **Dropout (0.5)** for overfitting prevention.
    * **Early Stopping** to halt training when validation loss stabilizes.

## 📊 Results & Performance

The models were evaluated based on **F1-Score**, Accuracy, Precision, Recall, and ROC-AUC.

| Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.81** | **0.82** | **0.88** |
| SVM (Linear) | 0.80 | 0.81 | 0.87 |
| Deep Learning (Keras) | 0.79 | 0.80 | 0.86 |
| Random Forest | 0.78 | 0.79 | 0.85 |

*Note: While the Deep Learning model was robust, Logistic Regression performed slightly better due to the dataset size (~2,800 samples), which favors linear models over complex neural networks.*

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/turkish-sentiment-analysis.git](https://github.com/yourusername/turkish-sentiment-analysis.git)
    ```

2.  **Install requirements:**
    ```bash
    pip install pandas numpy scikit-learn nltk tensorflow google-play-scraper TurkishStemmer matplotlib seaborn
    ```

3.  **Run the Notebook:**
    Open `project.ipynb` in Jupyter Notebook or Google Colab and run all cells sequentially. The script will:
    * Scrape fresh data.
    * Preprocess and train models.
    * Output performance metrics and graphs.

## 📈 Visualizations
The project includes:
* **Learning Curves:** To diagnose overfitting/underfitting.
* **Confusion Matrix:** To analyze false positives and false negatives.

## 📜 License
This project is for educational purposes.

---
**Author:** Habibe and Fatmanur
**Date:** January 2026
