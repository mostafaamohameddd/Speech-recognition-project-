import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import time

# === Settings ===
MODEL_PATH = "C:\\Users\\pc\\Speech real time\\model_compatible.h5"
SAMPLE_RATE = 16000
DURATION = 1  # seconds
N_MFCC = 13
MAX_LEN = 32

# === Load model and label map ===
model = load_model(MODEL_PATH)
label_map = {0: "go", 1: "no", 2: "off", 3: "stop", 4: "yes"}  # Match your model

# === Loop for repeated predictions ===
while True:
    # Countdown
    print("\n🕒 Get ready... Recording will start in 3 seconds")
    time.sleep(1)
    print("3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1... 🎙️  Start speaking NOW!")

    # Record audio
    audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    print("✅ Recording complete.")
    write("test.wav", SAMPLE_RATE, audio)

    # Process audio
    y, sr = librosa.load("test.wav", sr=SAMPLE_RATE)

    if len(y) < SAMPLE_RATE:
        y = np.pad(y, (0, SAMPLE_RATE - len(y)))
    else:
        y = y[:SAMPLE_RATE]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T

    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN, :]

    mfcc = mfcc[np.newaxis, ..., np.newaxis] #Add dimension

    # Predict
    prediction = model.predict(mfcc)
    label_index = np.argmax(prediction)
    confidence = np.max(prediction)
    predicted_command = label_map[label_index]

    print(f"🧠 Predicted command: {predicted_command} ({confidence:.2%} confidence)")

    # Ask if user wants to continue
    choice = input("\n🔁 Press Enter to try again or type 'q' to quit: ")
    if choice.strip().lower() == 'q':
        print("👋 Exiting.")
        break
