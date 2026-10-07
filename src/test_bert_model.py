import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "results/models/bert_reddit/final_model"

print("Loading saved BERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

text = (
    "I have been feeling very low lately and "
    "nothing seems enjoyable anymore."
)

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128,
)

inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

predicted_id = int(torch.argmax(probabilities).item())
predicted_label = model.config.id2label[predicted_id]
confidence = float(probabilities[predicted_id].item()) * 100

print("\nPrediction test completed.")
print(f"Predicted class: {predicted_label}")
print(f"Model confidence: {confidence:.2f}%")
print(f"Device: {device}")
