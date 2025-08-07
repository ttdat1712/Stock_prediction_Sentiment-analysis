# 📈 Stock Price Prediction using Deep Learning and Sentiment Analysis

This project aims to predict stock prices for **Apple** and **Amazon** by integrating historical stock data with Twitter sentiment analysis. It leverages **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** neural networks for predictive modeling and employs **Natural Language Processing (NLP)** techniques to extract market sentiment from tweets.

---

## 🧠 Project Overview

The project combines historical stock price data with social media sentiment to enhance stock price predictions. The key tasks include:

1. **Data Loading and Preprocessing**: Collect and preprocess historical stock data for Apple and Amazon.
2. **Sentiment Analysis**: Analyze Twitter data to derive sentiment scores related to Apple and Amazon stocks.
3. **Data Integration**: Combine sentiment scores with stock data to create enriched input features for predictive models.
4. **Model Training**: Train deep learning models (LSTM and GRU) to predict stock prices.
5. **Evaluation and Visualization**: Assess model performance and visualize predictions for future stock prices.

---

## 📂 Directory Structure
```
📁 project_root/
│
├── 📁 Amazon Stock Tweet Analysis
│   └── Amazon_Stock_Project.ipynb    # Jupyter notebook for Amazon stock analysis
├── 📁 Apple Stock Tweet Analysis
│   └── Stock_Project_Apple.ipynb     # Jupyter notebook for Apple stock analysis
├── requirements.txt                  # Project dependencies
└── README.md                         # Project overview and setup instructions
```
---

## 📊 Features Used
- Open, High, Low, Close, Volume
- Sentiment Scores from Twitter (using NLTK VADER)
- Technical indicators
  
---

## 🔧 Models Use
- LSTM
- GRU
- Bidirectional LSTM
- CNN + LSTM
All models were implemented using **TensorFlow Keras**.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Dependencies listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ttdat1712/Stock_prediction_Sentiment-analysis.git
   cd Stock_prediction_Sentiment-analysis
   ```
2. Install the required dependencies:
The project relies on the following Python libraries (listed in requirements.txt):
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- tensorflow and keras: Deep learning models (LSTM and GRU)
- nltk or vaderSentiment: NLP for sentiment analysis
- matplotlib or seaborn: Data visualization
- scikit-learn: Machine learning utilities
   ```bash
   pip install -r requirements.txt
   ```
3. Running the Project
- Open Amazon_Stock_Project.ipynb for Amazon stock analysis.
- Open Stock_Project_Apple.ipynb for Apple stock analysis.
- Follow the instructions in each notebook to preprocess data, train models, and visualize results.

---

## 📊 Methodology
1. Data Collection:
Search and collect data on Kaggle.
2. Sentiment Analysis:
- Tweets are processed using NLP tools to compute sentiment scores.
- Sentiment scores are aggregated to reflect daily or periodic market sentiment.
3. Data Preprocessing:
- Stock data is normalized and prepared for time-series analysis.
- Sentiment scores are aligned with stock data timestamps.
4. Model Training:
- LSTM and GRU models are trained on the combined dataset (stock prices + sentiment scores).
- Hyperparameters are tuned to optimize prediction accuracy.
5. Evaluation:
- Models are evaluated using metrics like Root Mean Squared Error (RMSE).
- Predictions are visualized to compare actual and predicted stock prices.

---

## 📈 Result
The project demonstrates the impact of combining sentiment analysis with historical data for stock price prediction.
Visualizations include:
- Time-series plots of stock prices and sentiment scores.
- Predicted vs. actual stock price comparisons.
- Model performance metrics.

---

## 🤖 Future Work
- Integrate real-time tweet scraping
- Try Transformer-based models like BERT for sentiment
- Deploy with Streamlit or Flask
