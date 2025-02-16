import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

# Preprocess data
def preprocess_data(data, text_column, label_column):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    labels = {"positive": 2, "neutral": 1, "negative": 0}
    data['label'] = data[label_column].map(labels)

    # Remove rows with empty or invalid text entries
    data = data[data[text_column].notnull() & (data[text_column].str.strip() != "")]

    # Tokenize the text column
    print("Tokenizing data...")
    tokenized = tokenizer(
        list(data[text_column]),
        padding="max_length",
        truncation=True,
        return_tensors="pt"  # PyTorch tensors
    )

    labels = torch.tensor(data['label'].values)  # Convert labels to tensor
    return tokenized, labels

# Define PyTorch Dataset class
class SentimentDataset(Dataset):
    def __init__(self, tokenized_data, labels):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Train the model
def train_model(train_loader, val_loader, model, optimizer, device, epochs=3):
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Training Loss: {total_loss / len(train_loader):.4f}")

        evaluate_model(val_loader, model, device)

# Evaluate the model
def evaluate_model(val_loader, model, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

# Main script
if __name__ == "__main__":
    # Load the dataset
    data_path = "combined_dataset.csv"
    try:
        data = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(data_path, encoding='latin1')

    # Define text and label columns
    text_column = "text"
    label_column = "sentiment"

    # Preprocess the data
    tokenized_data, labels = preprocess_data(data, text_column, label_column)

    # Convert tokenized_data to lists for compatibility
    input_ids = tokenized_data['input_ids']
    attention_mask = tokenized_data['attention_mask']

    # Split each component and the labels
    train_input_ids, val_input_ids, train_attention_mask, val_attention_mask, train_labels, val_labels = train_test_split(
        input_ids, attention_mask, labels, test_size=0.2, random_state=42
        )

    # Create dictionaries for training and validation datasets
    train_inputs = {'input_ids': train_input_ids, 'attention_mask': train_attention_mask}
    val_inputs = {'input_ids': val_input_ids, 'attention_mask': val_attention_mask}

    # Create Dataset and DataLoader
    train_dataset = SentimentDataset(train_inputs, train_labels)
    val_dataset = SentimentDataset(val_inputs, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize the model and optimizer
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train and evaluate the model
    train_model(train_loader, val_loader, model, optimizer, device, epochs=3)

    # Save the model
    model_path = "/content/drive/My Drive/sentiment_model2"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

   