from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Replace 'distilbert-base-uncased' with your desired model
model_name = "distilbert-base-uncased"

# Download the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Save the model locally
model.save_pretrained("./local_model")
tokenizer.save_pretrained("./local_model")
