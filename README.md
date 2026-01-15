# IMDB Sentiment Analysis RNN

🎬 Deep Learning–based Sentiment Analysis to classify IMDB movie reviews as **positive** or **negative** using a **Recurrent Neural Network (RNN)**.

---

## 📌 Project Description
This project applies Natural Language Processing (NLP) and RNNs to learn patterns in text and predict sentiment from IMDB reviews. It demonstrates skills in text preprocessing, model training, evaluation, and exporting a trained model for use.

---

## 🚀 Features
- Tokenization and padding of text sequences
- Word embeddings
- Simple RNN model
- Flask/Streamlit-based interface (app.py) to test live inputs
- Trained model included

---

## 🛠️ Tech Stack
- **Python**
- **TensorFlow/Keras**
- **NumPy**
- **Pandas**
- **NLTK**
- **Jupyter Notebook**
- **Streamlit / Flask (optional)**

---

## 📂 Project Structure
```text
IMDB-Sentiment-Analysis-RNN/
├── Sentiment_annalysis_using_RNN_.ipynb   # Core notebook
├── app.py                                 # Web/terminal interface
├── imdb_rnn_model.h5                      # Trained model
├── requirements.txt                       # Dependencies
└── README.md                              # Project documentation

🧪 Model Results
| Evaluation     | Score   |
| -------------- | ------- |
| Train Accuracy | ~90%    |
| Test Accuracy  | ~85–88% |

🧠 How to Run

1.Clone the repo:
git clone https://github.com/amritgupta0680/IMDB-Sentiment-Analysis-RNN.git

2.Install dependencies:
pip install #required libraries

3.Run the notebook:
jupyter notebook

4.Run the app:
streamlit run app.py

📊 Future Improvements

Replace RNN with LSTM/GRU

Add attention or pretrained embeddings (e.g., GloVe/BERT)

Deploy as web app

Add unit tests

👤 Author

Amrit Gupta
GitHub: https://github.com/amritgupta0680
