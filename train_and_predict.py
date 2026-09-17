import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# 1. Dataset Setup
data = {
    "text": [
        "The application crashes every time I click on submit button.",
        "System keeps freezing when opening settings panel.",
        "The checkout page throws a 500 error when I enter my card details.",
        "App closes unexpectedly after the latest software patch.",
        "Getting a null pointer exception on the login screen.",
        "Can you please add a dark mode option in the UI?",
        "Feature request: Allow export to CSV format.",
        "It would be amazing if we could export reports directly to PDF format.",
        "Please implement Google Single Sign-On for faster login.",
        "Would love an offline reading feature for saved documents.",
        "Really loving the new responsive UI update! Great work.",
        "Super smooth performance and fast navigation.",
        "The user interface is extremely clean and intuitive. Loving it!",
        "Fantastic experience using this service, highly recommended!",
        "Best software tool I have used this entire year.",
        "This update is terrible and very slow.",
        "Extremely disappointed with the customer service.",
        "This is worst update ever. The app takes 2 minutes just to open.",
        "Ugly design and frustrating user navigation layout.",
        "Extremely laggy experience, completely unusable right now."
    ],
    "target": [0, 0, 0, 0, 0,  1, 1, 1, 1, 1,  2, 2, 2, 2, 2,  3, 3, 3, 3, 3]
}

raw_dataset = Dataset.from_dict(data)

# 2. Tokenization & Model Setup
id2label = {0: "Bug Report", 1: "Feature Request", 2: "Positive Feedback", 3: "Negative Feedback"}
label2id = {v: k for k, v in id2label.items()}

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("target", "label")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    id2label=id2label,
    label2id=label2id
)

# 3. Fine-Tuning Setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=12,
    per_device_train_batch_size=4,
    logging_steps=2,
    learning_rate=3e-5,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 4. Execute Fine-Tuning
print("Starting DistilBERT Fine-Tuning...")
trainer.train()

# 5. Save Model & Tokenizer Weights Locally
model.save_pretrained("./distilbert_intent_classifier")
tokenizer.save_pretrained("./distilbert_intent_classifier")
print("Model & Tokenizer saved successfully!")

# 6. Run Sample Predictions
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_sentences = [
    "The payment screen throws an error and crashes.",
    "Can you add option to filter reports by custom date?",
    "Awesome platform! The speed improvement is noticeable.",
    "Total waste of time, app is unusable and full of bugs."
]

print("\n=== SAMPLE MODEL PREDICTION TEST SUMMARY ===\n")

for text in test_sentences:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)[0]
        
    predicted_id = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_id].item() * 100
    predicted_label = id2label[predicted_id]
    
    print(f"Input Text : '{text}'")
    print(f"Prediction : {predicted_label} ({confidence:.2f}% Confidence)")
    print("-" * 60)
