# SMS Spam Detection using NLP

This project implements an SMS spam detection system using Natural Language Processing (NLP) and machine learning. The system classifies SMS messages as either spam or ham (legitimate) using a Naive Bayes classifier with TF-IDF features.

## Features

- Preprocesses text data (tokenization, stopword removal, lemmatization)
- Trains a machine learning model to classify messages
- Provides a simple REST API for spam detection
- Clean and responsive web interface

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sms-spam-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   - Download the SMS Spam Collection dataset from [here](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
   - Place the `SMSSpamCollection` file in the project root directory

## Usage

### 1. Train the Model

Run the following command to train the spam detection model:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train a Multinomial Naive Bayes classifier
- Save the trained model and vectorizer to disk
- Print evaluation metrics on the test set

### 2. Start the Web Application

Run the Flask application:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### 3. Using the Web Interface

1. Open `sms.html` in a web browser
2. Enter or paste an SMS message in the text area
3. Click "Check for Spam" to analyze the message
4. View the results indicating whether the message is spam or ham

### API Endpoints

- `GET /`: Check if the API is running
- `POST /detect`: Detect if a message is spam
  - Request body: `{"message": "Your message here"}`
  - Response: 
    ```json
    {
        "is_spam": true,
        "probability": 0.95,
        "message": "SPAM detected!"
    }
    ```

## Project Structure

- `app.py`: Flask web application
- `train_model.py`: Script to train and save the spam detection model
- `sms.html`: Web interface for the spam detector
- `requirements.txt`: Python dependencies
- `spam_detection_model.joblib`: Trained model (created after training)
- `tfidf_vectorizer.joblib`: Fitted TF-IDF vectorizer (created after training)

## Model Performance

The model's performance on the test set will be displayed after training. Typical metrics include:

- Accuracy: >98%
- Precision (spam): >95%
- Recall (spam): >85%
- F1-score: >90%

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- Built with Python, Flask, scikit-learn, and NLTK
