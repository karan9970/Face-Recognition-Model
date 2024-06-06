# Face Recognition Model
# Dataset : https://www.kaggle.com/datasets/trainingdatapro/age-detection-human-faces-18-60-years
This repository contains the code for a face recognition model using TensorFlow and MobileNetV2. The model is trained to recognize faces from a dataset organized in training and testing directories. 

## Project Structure

- `train/`: Directory containing training images, organized in subdirectories named after each class/person.
- `test/`: Directory containing testing images, organized in subdirectories named after each class/person.
- `age_detection.csv`: (Optional) CSV file containing additional metadata for the dataset.
- `face_recognition.py`: Python script to train, evaluate, and make predictions with the face recognition model.
- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: Project documentation.

## Requirements

- Python 3.6 or higher
- TensorFlow
- Pandas
- Numpy

## Setup

1. Clone the repository:

   ```sh
   git clone https://github.com/your-username/face-recognition-model.git
   cd face-recognition-model
