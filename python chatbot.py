import difflib
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
model = tf.keras.models.load_model(HERE / "healthsure_nn.keras")
scaler = joblib.load(HERE / "scaler.joblib")
with open(HERE / "meta.json", encoding="utf-8") as f:
    meta = json.load(f)
CLASSES, VOCAB = meta["classes"], meta["vocab"]

class Quit(Exception):
    pass


def ask_number(prompt, lo, hi):
    """Ask until the user gives a number between lo and hi."""
    while True:
        text = input(f"Bot: {prompt} ").strip()
        if text.lower() in ("quit","exit","q"):
            raise Quit
        try:
            value = float(text)
        except ValueError:
            print("Bot: Please enter a number.")
            continue
        if lo <= value <= hi:
            return value
        print(f"Bot: That looks out of range ({lo}-{hi}). Please check and retry.")

def match_symptom(text):
    """Map free text to known symptons. Returns (matched, unmatched )."""
    matched, unmatched = [], []
    for raw in text.lower().replace(";",",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        if raw in VOCAB:
            hit = raw
        else:
            subs = [v for v in VOCAB if raw in v or v in raw]
            close = difflib.get_close_matches(raw, VOCAB, n=1, cutoff=0.6)
            hit = subs[0] if subs else (close[0] if close else None)
        if hit and hit not in matched:
            matched.append(hit)
        elif not hit:
            unmatched.append(raw)
    return matched, unmatched 

def predict(age, height, weight, hb, wbc, plt, symptoms):
    bmi = weight / (height / 100) ** 2
    
    values = {
        "Age_years": age,
        "Height_cm": height,
        "BMI": bmi,
        "Hemoglobin_g_dL": hb,
        "WBC_10^3_per_uL": wbc,
        "Platelets_10^3_per_uL": plt
    }

    nums = np.array(
        [[values[c] for c in meta["num_cols"]]],
        dtype="float32"
    )

    sym = np.array([[1.0 if v in symptoms else 0.0 for v in VOCAB]], dtype="float32")
    x = np.hstack([scaler.transform(nums),sym]).astype("float32")
    probs = model.predict(x, verbose=0)[0]
    top = probs.argsort()[::-1][:3]
    return [(CLASSES[i], float(probs[i])) for i in top],bmi

def one_consultation():
    age = ask_number("How old is the patient (years)?", 0, 120)
    height = ask_number("Height in cm?", 50, 250)
    weight = ask_number("weight in kg?", 10, 300)
    hb = ask_number("Hemoglobin (g/dL)?", 3, 20)
    wbc = ask_number("WBC count(x10^3/uL)?", 0.5, 100)
    plt = ask_number("Platelets (x10^3/uL)?", 5, 1000)

    print("Bot: Known symptoms I understand:")
    print("    "+";".join(VOCAB))
    while True:
        text = input("Bot: List the symptoms seperated by commas:").strip()
        if text.lower() in ("quit", "exit", "q"):
            raise Quit 
        symptoms, unknown = match_symptom(text)
        if unknown:
            print(f"Bot: I did not recognise:{','.join(unknown)}")
        if symptoms:
            print(f"Bot: using symptoms:{','.join(symptoms)}")
            break
        print("Bot: I need at least one recognised symptom from the list above.")

    top, bmi = predict(age, height, weight, hb, wbc, plt, symptoms)
    print(f"\nBot: (BMI calculated: {bmi:.1f})")
    print("Bot: The model's most likely recorded-diagnosis labels:")
    for name, p in top:
        print(f"      _ {name}: {p:.0%})")
    print("\nBot: REMINDER: this model was trained on 50 synthetic records for")
    print("     software testing. It is not a diagnosis. For real health")
    print("     concerns, please see a qualified doctor.\n")


def main():
    print("=" * 60)
    print(" HealthSure test chatbot (synthetic data - not medical advice)")
    print("=" * 60)
    try:
        while True:
            one_consultation()
            again = input("Bot: Check another patient? (y/n) ").strip().lower()
            if again not in ("y", "yes"):
                break
    except (Quit, KeyboardInterrupt, EOFError):
        pass
    print("Bot: Goodbye!")


if __name__ == "__main__":
    main()
