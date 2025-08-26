# music_genre_classification
This project uses CNN to classify music genres.
Music Genre Classification using Deep Learning

Project Overview

This project demonstrates how Convolutional Neural Networks (CNNs) can classify music into genres by analyzing spectrograms of audio signals.
Instead of using handcrafted features (like MFCC averages), I converted audio into 2D spectrogram “images”, allowing CNNs to detect time-frequency patterns.

✅ Built using Python, Librosa, TensorFlow/Keras
✅ Dataset: GTZAN Music Genre Dataset
 (10 genres, 1000 songs, 30s each)
✅ Achieved 43% test accuracy

🛠️ Tech Stack

Python 🐍

Librosa (audio preprocessing, spectrograms)

OpenCV / NumPy (image handling)

TensorFlow / Keras (deep learning model)

Matplotlib / Seaborn (data visualization)

📂 Dataset

10 Genres: Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock

~100 audio files per genre (.wav, 30 seconds each)

Each audio file → converted into a 128×128 Mel-Spectrogram image

🔎 Approach

Preprocessing

Loaded .wav files (30s clips)

Converted to Mel Spectrograms using Librosa

Resized to 128×128 grayscale images

Normalized pixel values (0–1)

Saved dataset as .npy for faster re-runs

Modeling

Built a Convolutional Neural Network (CNN)

Architecture:

Conv2D(32) → MaxPool  
Conv2D(64) → MaxPool  
Flatten → Dense(128) → Dropout  
Dense(10, softmax)


Loss: categorical_crossentropy

Optimizer: Adam

Metrics: accuracy

Training & Evaluation

Train/Test Split: 80/20

Epochs: 20

Batch size: 32

Tracked accuracy/loss curves

Evaluated with confusion matrix & classification report

📊 Results

Test Accuracy: 43%

Confusion Matrix:

Sample Spectrogram:

Accuracy Curve:

🚀 How to Run

Clone the repo:

git clone https://github.com/Simrozechawla/music_genre_classification.git
cd music-genre-classification


Install dependencies:

pip install -r requirements.txt


Run preprocessing (creates X.npy, y.npy):

python preprocess.py


Train model:

python train.py


Evaluate:

python evaluate.py

🎤 Future Improvements

Add data augmentation (time-shift, noise, pitch-shift)

Try advanced architectures (ResNet, CNN-LSTM hybrids)

Deploy as a Streamlit Web App → Upload a song → Predict genre

📌 Learnings

CNNs can learn from spectrograms like they learn from images

Audio preprocessing is as important as the model

Saving preprocessed data speeds up experimentation significantly

Model deployment adds huge portfolio value

👤 Author

Simroze Chawla
🎓 Electronics & Communication Engineering (ECE) Student
📍 TIET, Patiala
💼 Aspiring VLSI & AI/ML Engineer


🏷️ Tags

#DeepLearning #MachineLearning #AudioProcessing #CNN #ECE #AIProjects
