## Real-Time Speech Command Recognition



A lightweight, real-time speech recognition system that classifies five common spoken commands using a Convolutional Neural Network (CNN) trained on the TensorFlow Speech Commands dataset. This project enables hands-free interaction for applications like voice-controlled devices or simple IoT integrations.

## Table of Contents





Features



Demo



Prerequisites



Installation



Usage



Training the Model



Project Structure



Contributing



License



Acknowledgments


## Features





Real-Time Audio Capture: Records 1-second audio clips from your microphone using sounddevice.



MFCC Feature Extraction: Processes audio into Mel-Frequency Cepstral Coefficients (MFCCs) for efficient input to the CNN.



Command Classification: Recognizes five commands: go, stop, yes, no, off with confidence scores.



Modular Design: Separate scripts for inference (real_time.py) and training (Speech recognition model training.ipynb).



Cross-Platform: Tested on Windows, macOS, and Linux (requires microphone access).


## Demo
Here's a quick simulation of the output:

 Get ready... Recording will start in 3 seconds
3...
2...
1...   Start speaking NOW!
 Recording complete.
 Predicted command: yes (92.3% confidence)

 Press Enter to try again or type 'q' to quit:

## Prerequisites





Python 3.8 or higher



A microphone-enabled device



Access to Google Colab for model training (optional, if using the provided notebook)


## Installation

1) Clone the repository:
git clone https://github.com/yourusername/speech-command-recognition.git
cd speech-command-recognition

2) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3)Install dependencies:

pip install -r requirements.txt



## Training the Model

The CNN model is trained on a subset of the TensorFlow Speech Commands v0.02 dataset, focusing on the five target commands.





Open Speech recognition model training.ipynb in Google Colab.



Run the cells sequentially:





Mount Google Drive for data persistence.



Download and extract the dataset.



Preprocess and split data (70% train, 15% validation, 15% test).



Build and train the CNN (uses TPU acceleration for speed).



Evaluate with classification report and confusion matrix.



Save the model as model_compatible.h5 (HDF5 format, optimizer excluded for compatibility).

Model Architecture Highlights:





Input: MFCC features (13 coefficients, max length 32).



Layers: Conv1D + MaxPooling1D, followed by Dense layers.



Output: Softmax for 5-class classification.



Achieved ~95% accuracy on test set (results may vary).

For local training, adapt the notebook to run in Jupyter.


## Project Structure

speech-command-recognition/
├── .gitignore              # Ignores temp files, caches, and env vars
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── real_time.py            # Real-time inference script
├── model_compatible.h5     # Trained CNN model (train your own)
└── Speech recognition model training.ipynb  # Training notebook (Colab-ready)

