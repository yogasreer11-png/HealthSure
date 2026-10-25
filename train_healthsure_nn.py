import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler 
HERE = Path(__file__).resolve().parent
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
FILE = r"C:\Users\Yoga\Documents\pytorch ai bot\HealthSure_Synthetic_Cancer_Test_Data_50.xlsx"
NUM_COLS = ["Age_years","Height_cm","BMI","Hemoglobin_g_dL","WBC_10^3_per_uL","Platelets_10^3_per_uL",]

df = pd.read_excel(FILE, sheet_name="Synthetic_Test_Data")
df["BMI"] = df["Weight_kg"]/(df["Height_cm"]/100)**2

sym_lists = df["Reported_Symptoms"].str.lower().str.split(";").apply(lambda xs:[s.strip()for s in xs])
vocab = sorted({s for xs in sym_lists for s in xs})
sym = np.array([[1.0 if v in xs else 0.0 for v in vocab] for xs in sym_lists])

X_num = df[NUM_COLS].values.astype("float32")
le = LabelEncoder()
y = le.fit_transform(df["Recorded_Diagnosis"])
n_classes = len(le.classes_)
print(f"{len(df)}rows|{n_classes}classes|{len(vocab)}symptomfeatures")
def build_model(n_features):
    model = tf.keras.Sequential([
       tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(32,activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16,activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.Dense(n_classes,activation="softmax"), ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def make_features(scaler,X_num_part,sym_part,fit=False):
    nums=scaler.fit_transform(X_num_part) if fit else scaler.transform(X_num_part)
    return np.hstack([nums,sym_part]).astype("float32")

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="loss",patience=30,restore_best_weights=True)


skf = StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
accs = []
for fold,(tr,te) in enumerate(skf.split(X_num,y),1):
    scaler =   StandardScaler()
    Xtr = make_features(scaler,X_num[tr],sym[tr],fit=True)
    Xte = make_features(scaler,X_num[te],sym[te])
    model = build_model(Xtr.shape[1])
    model.fit(Xtr,y[tr],epochs=300,batch_size=8,verbose=0,callbacks=[early_stop])
    _,acc= model.evaluate(Xte,y[te],verbose=0)
    accs.append(acc)
    print(f"Fold {fold}: held-out accuracy = {acc:.2%}")
print(f"Mean cross-validated accuracy:{np.mean(accs):.2%}(+/-{np.std(accs):.2%})")

scaler =   StandardScaler()
X_all = make_features(scaler,X_num,sym,fit=True)
final = build_model(X_all.shape[1])
final.fit(X_all,y,epochs=300,batch_size=8,verbose=0,callbacks=[early_stop])
final.save(HERE/"healthsure_nn.keras")
joblib.dump(scaler, HERE / "scaler.joblib")
with open(HERE / "meta.json", "w", encoding="utf-8") as f:
    json.dump({"classes": list(le.classes_),
               "vocab": vocab,
               "num_cols": NUM_COLS}, f, indent=2)
print("Saved: healthsure_nn.keras, scaler.joblib, meta.json")
print("Now run:  python chatbot.py")
 
