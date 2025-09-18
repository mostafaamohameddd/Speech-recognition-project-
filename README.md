# Real-Time Speech Recognition

This project recognizes 5 spoken commands in real-time using a trained CNN model.

**Supported Commands:**  
`go`, `stop`, `yes`, `no`, `off`

## Files

- `real_time.py`: Runs live speech prediction
- `train_model.ipynb`: Model training code (Colab)
- `model_compatible.h5`: Trained model (not uploaded)
- `requirements.txt`: Dependencies

## Run

Install dependencies:
pip install -r requirements.txt

Then run:
python real_time.py

## Coming Soon

- Web-based deployment (streamlit or Flask)
