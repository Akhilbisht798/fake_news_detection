import tkinter as tk
import joblib

vectorizer_loaded, model_loaded = joblib.load("svm_model.pkl")

def classify_text():
    input_text = entry.get()
    text = vectorizer_loaded.transform([input_text])
    predict = model_loaded.predict(text)
    result_label.config(text="FAKE" if predict == 0 else "NOT FAKE")



window = tk.Tk()
window.title("Fake News Detection")
window.geometry("800x500")

# Create the input entry widget
entry = tk.Entry(window, width=80)
entry.pack(pady=50)

# Create the classify button
classify_button = tk.Button(window, text="Classify", command=classify_text)
classify_button.pack()

# Create the label to display the result
result_label = tk.Label(window, text="")
result_label.pack(pady=50)

# Start the GUI event loop
window.mainloop()