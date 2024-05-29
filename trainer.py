import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentiment_dataset import SentimentDataset
from model_callback import SaveBestModelCallback
from config import get_args

args = get_args()
class Trainer:
    def __init__(self, model_name, device, batch_size, num_epochs):
        self.model_name = args.model_name
        self.device = args.device
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=args.num_labels)
        self.model = self.model.to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

    def train(self, X_train, y_train, X_val, y_val):
        train_data = self.tokenizer(list(X_train), return_tensors="pt", padding=True, truncation=True)
        valid_data = self.tokenizer(list(X_val), return_tensors="pt", padding=True, truncation=True)

        train_dataset = SentimentDataset(
            input_ids=train_data['input_ids'],
            attention_mask=train_data['attention_mask'],
            labels=y_train
        )

        val_dataset = SentimentDataset(
            input_ids=valid_data['input_ids'],
            attention_mask=valid_data['attention_mask'],
            labels=y_val
        )

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        save_callback = SaveBestModelCallback(self.model, '/content/best_model.pth')

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            progress_bar = tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader))
            for step, batch in progress_bar:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = self.criterion(outputs.logits, batch['labels'])
                loss.backward()
                self.optimizer.step()

                predictions = torch.argmax(outputs.logits, dim=1)
                correct = (predictions == batch['labels']).sum().item()
                total_correct += correct
                total_samples += len(batch['labels'])

                total_loss += loss.item()

                progress_bar.set_description(f'Epoch {epoch+1}/{self.num_epochs}, Step {step}/{len(train_dataloader)}')
                progress_bar.set_postfix({'Loss': loss.item(), 'Accuracy': correct / len(batch['labels'])})

            epoch_loss = total_loss / len(train_dataloader)
            epoch_accuracy = total_correct / total_samples

            self.model.eval()
            val_total_loss = 0
            val_total_correct = 0
            val_total_samples = 0

            with torch.no_grad():
                for val_step, val_batch in enumerate(val_dataloader, 1):
                    val_batch = {key: value.to(self.device) for key, value in val_batch.items()}
                    val_outputs = self.model(**val_batch)
                    val_loss = self.criterion(val_outputs.logits, val_batch['labels'])

                    val_predictions = torch.argmax(val_outputs.logits, dim=1)
                    val_correct = (val_predictions == val_batch['labels']).sum().item()
                    val_total_correct += val_correct
                    val_total_samples += len(val_batch['labels'])

                    val_total_loss += val_loss.item()

            val_epoch_loss = val_total_loss / len(val_dataloader)
            val_epoch_accuracy = val_total_correct / val_total_samples

            save_callback(val_epoch_accuracy)

            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}')
