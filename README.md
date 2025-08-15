# 📰 Fake News Detection with Machine Learning

This project uses **Natural Language Processing (NLP)** and **machine learning** to classify news articles as **fake** or **real**. It includes data preprocessing, feature extraction, model training, and evaluation on separate datasets.

## ✨ Features
- **Classification Models**: Logistic Regression and SGD Classifier
- **Feature Extraction**: TF-IDF vectorization
- **Evaluation Metrics**: Confusion matrix, ROC curve, and classification report
- **Interactive Notebooks**: Easy-to-run Jupyter Notebooks for training and testing
- **Saved Model**: Pre-trained model for quick predictions


## 📊 Dataset
- **Full Dataset Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

## ⚙️ Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. Download full datasets from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

## 🚀 Usage
1. **Run the Training Notebook**:
   ```bash
   jupyter notebook FakeNewsDetect.ipynb
   ```
   This notebook handles data preprocessing, model training, and evaluation.

2. **Run the Testing Notebook**:
   ```bash
    jupyter notebook FakeNewsTryout.ipynb
   ```
   Use this to test the model on unseen data.

3. **Predict with the Saved Model**:
   ```python
   import joblib
   model = joblib.load("models/fake_news_model.pkl")
   vectorizer = joblib.load("models/fake_news_vectorizer.pkl")
   text = ["Breaking: Scientists discover AI that writes perfect README files!"]
   x = vectorizer.transform(text)
   prediction = model.predict(x)
   print("Prediction:", "Fake" if prediction[0] == 1 else "Real")
   ```

## 📈 Example Results
- **Accuracy**: ~95% on unseen dataset
- **Classification Report**:
  ```
                precision    recall  f1-score   support
  Real          0.99        0.93     0.95      4900
  Fake          0.93        0.99     0.96      5000
  ```
- **Confusion Matrix**: Visualized in `FakeNewsDetect.ipynb`

## 🔮 Future Work
- Upgrade to transformer-based models (e.g., BERT)
- Deploy as a web app using Flask or Streamlit
- Develop a real-time fake news detection API

## 📜 License
This project is licensed under the [MIT License](LICENSE).

## 🤝 Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please open an issue to discuss major changes before submitting a pull request.

## 📧 Contact
- **Author**: KESABA-BARIK
- **Email**: kesababarik007@gmail.com
- **GitHub**: [KESABA-BARIK](https://github.com/KESABA-BARIK)
