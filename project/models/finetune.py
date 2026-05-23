import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

class FineTuner:
    def __init__(self, model, train_loader, val_loader, class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.best_model_state = None
        self.best_val_acc = 0.0
        self.class_weights = class_weights
        
        print(f"Device: {self.device}")
    
    def _train_epoch(self, optimizer, criterion):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(self.train_loader, desc="Train"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = (running_corrects.double() / len(self.train_loader.dataset)).cpu().item()
        return epoch_loss, epoch_acc
    
    def _validate(self, criterion):
        self.model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Val"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = val_loss / len(self.val_loader.dataset)
        epoch_acc = (val_corrects.double() / len(self.val_loader.dataset)).cpu().item()
        return epoch_loss, epoch_acc
    
    def fit(self, epochs, learning_rate=0.001, weight_decay=1e-4):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 30)
            
            train_loss, train_acc = self._train_epoch(optimizer, criterion)
            val_loss, val_acc = self._validate(criterion)
            
            scheduler.step(val_acc)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = deepcopy(self.model.state_dict())
                print(f"New best model! Val Acc: {self.best_val_acc:.4f}")
        
        # Load model terbaik setelah training selesai
        self.model.load_state_dict(self.best_model_state)
        print(f"\nTraining complete! Best Val Acc: {self.best_val_acc:.4f}")
        return self.model
    
    def plot_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        plt.show()
    
    def save_model(self, path):
        torch.save(self.best_model_state, path)
        print(f"Model saved to {path}")