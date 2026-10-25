HealthSure

A TensorFlow/Keras neural network and terminal chatbot trained on a synthetic 50-record dataset. It predicts a recorded cancer-type label from basic patient details, symptoms and blood counts.

⚠️ Not a medical tool. HealthSure is a software-testing and learning project. It is trained on fictional data and cannot diagnose cancer. Do not use it for real health decisions. If you have health concerns, see a qualified doctor.

Features
Feed-forward neural network (dense layers, dropout, L2 regularization) built with TensorFlow/Keras
Stratified 3-fold cross-validation, since the dataset has only 3–4 records per class
Saves the trained model, scaler and metadata for reuse
Interactive terminal chatbot with top-3 predictions and confidence scores
Fuzzy matching so symptoms can be typed in plain words
How it works
Item	Details
Target	Recorded_Diagnosis (15 cancer types)
Numeric inputs	Age, height, BMI (derived from height and weight), hemoglobin, WBC, platelets
Symptom input	Reported_Symptoms split on ; and multi-hot encoded (19 symptoms)
Dropped columns	Patient_ID (identifier); Diagnosis_Source, Data_Note, Result_Status (identical in every row)
Model	Dense(32, ReLU) → Dropout(0.3) → Dense(16, ReLU) → Dense(15, softmax)
Project structure
HealthSure/
├── train_healthsure_nn.py                        # trains and saves the model
├── chatbot.py                                    # terminal chatbot
├── HealthSure_Synthetic_Cancer_Test_Data_50.xlsx # synthetic dataset
├── healthsure_nn.keras                           # created after training
├── scaler.joblib                                 # created after training
└── meta.json                                     # created after training
Setup

TensorFlow needs a supported Python version (3.10–3.13 at the time of writing; check tensorflow.org/install).

bash
python -m venv tfenv
tfenv\Scripts\activate          # Windows
# source tfenv/bin/activate     # macOS / Linux
python -m pip install pandas numpy openpyxl scikit-learn tensorflow
Usage
bash
python train_healthsure_nn.py   # train, evaluate, save model files
python chatbot.py               # start the chatbot (type 'quit' to exit)

Example session:

Bot: How old is the patient (years)? 52
...
Bot: List the symptoms, separated by commas: persistent cough, weight loss
Bot: The model's most likely recorded-diagnosis labels:
     - Lung cancer: 78%
     ...
Limitations
Tiny dataset: 50 rows is too few for a neural network to generalize, so accuracy estimates are noisy.
Symptoms dominate: in this synthetic data the symptoms almost determine the label, so the blood values contribute little and confidence scores look overstated.
Fixed symptom list: only the 19 symptoms in the dataset are recognized.
Synthetic data: the model has learned nothing about real patients.
Possible next steps
Train on larger, properly sourced and anonymized data, including non-cancer cases
Add a GUI (Tkinter) or web interface (Streamlit / Gradio)
Try other model types and compare them against a simple baseline such as logistic regression
License
