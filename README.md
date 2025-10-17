# 🧠 Handwritten Digit Recognizer

This project focuses on building a machine learning model capable of recognizing handwritten digits (0–9) from grayscale 28×28 pixel images. The is the MNIST dataset and is provided in CSV format.

## 📂 Dataset Description

- Each image is represented by 784 pixel values (28×28) in a single row.
- `train.csv` — contains labeled training data.
- `label`: the correct digit (0–9).
- `pixel0` → pixel783: grayscale values from 0 (white) to 255 (black).
- `test.csv` — contains the same pixel structure, but without labels.
- The task is to train a model on train.csv and then predict the labels for test.csv.
- Each image can be reshaped into a 28×28 grid for visualization.

## ⚙️ Data Preprocessing

- Normalized pixel values to range [0, 1].
- Split training data into train/validation subsets.
- Optionally reshaped data to (28, 28, 1) for CNN models.

`X = train.drop('label', axis=1) / 255.0
y = train['label']`

## 🧠 Model Architecture
### 🔹 Fully Connected Neural Network (Baseline)
``Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])``

### 🔹 Convolutional Neural Network (Advanced)
`Sequential([ \n
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])`


- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

## 📊 Evaluation

- Performance metric: categorization accuracy
- A model with accuracy ≈ 0.99 means it correctly classifies 99% of test images.

## 🧾 Technologies Used

- Python 
- NumPy / Pandas 
- TensorFlow / Keras
- Matplotlib / Seaborn
- scikit-learn

## 📈 Results
### Model Type	Accuracy	Notes
### CNN	~0.99	Best performing model

## 💡 Future Improvements

- Use data augmentation to improve robustness.
- Experiment with dropout and batch normalization.
- Try ensemble models or advanced architectures like ResNet.

GitHub link:


# 🧑‍💻 Author

## Norik Hovhannisyan
Machine Learning & Data Science Enthusiast
