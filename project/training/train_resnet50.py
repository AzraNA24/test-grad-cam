import sys
import os
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import dari dataloader
try:
    from preprocessing.dataloader import train_loader, val_loader, test_loader, train_df
    print("DataLoaders loaded successfully")
except ImportError:
    print("ERROR: Please run dataloader.py first")
    sys.exit(1)

from models.transfer_learning import ResNet50TransferLearning
from models.finetune import FineTuner

def get_class_weights(train_df):
    """Hitung class weights untuk mengatasi imbalance dataset"""
    labels = train_df['label'].values
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

def evaluate_on_test(model, test_loader):
    """
    Evaluasi model di test set.
    Parameter model: model yang sudah dilatih (state terbaik)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\n" + "="*50)
    print("TEST SET EVALUATION RESULTS")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=['NORMAL', 'PNEUMONIA']))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix_test.png', dpi=150)
    plt.show()
    
    return all_preds, all_labels

def experiment_frozen():
    """Eksperimen 1: Hanya classifier head yang dilatih"""
    print("\n" + "="*60)
    print("EXPERIMENT 1: FROZEN BACKBONE")
    print("="*60)
    
    model = ResNet50TransferLearning(num_classes=2)
    weights = get_class_weights(train_df)
    
    finetuner = FineTuner(model, train_loader, val_loader, weights)
    finetuner.fit(epochs=7, learning_rate=0.001, weight_decay=1e-4)
    finetuner.plot_history()
    finetuner.save_model('resnet50_frozen.pth')
    
    # Evaluasi di test set menggunakan model yang sudah dilatih
    print("\nEvaluating frozen model on test set...")
    trained_model = finetuner.model  # model sudah dalam state terbaik
    evaluate_on_test(trained_model, test_loader)
    
    return finetuner

def experiment_unfreeze_layer4():
    """Eksperimen 2: Classifier head lalu unfreeze layer4"""
    print("\n" + "="*60)
    print("EXPERIMENT 2: UNFREEZE LAYER4")
    print("="*60)
    
    model = ResNet50TransferLearning(num_classes=2)
    weights = get_class_weights(train_df)
    
    # Phase 1: Classifier only
    print("\n--- PHASE 1: Classifier head (5 epochs, LR=0.001) ---")
    finetuner_stage1 = FineTuner(model, train_loader, val_loader, weights)
    finetuner_stage1.fit(epochs=5, learning_rate=0.001, weight_decay=1e-4)
    
    # Phase 2: Unfreeze layer4
    print("\n--- PHASE 2: Unfreeze layer4 (7 epochs, LR=0.0001) ---")
    model.unfreeze_layer4()
    finetuner_stage2 = FineTuner(model, train_loader, val_loader, weights)
    finetuner_stage2.fit(epochs=7, learning_rate=0.0001, weight_decay=1e-4)
    
    # History dari kedua phase
    finetuner_stage2.history = {
        'train_loss': finetuner_stage1.history['train_loss'] + finetuner_stage2.history['train_loss'],
        'train_acc': finetuner_stage1.history['train_acc'] + finetuner_stage2.history['train_acc'],
        'val_loss': finetuner_stage1.history['val_loss'] + finetuner_stage2.history['val_loss'],
        'val_acc': finetuner_stage1.history['val_acc'] + finetuner_stage2.history['val_acc']
    }
    finetuner_stage2.best_val_acc = max(finetuner_stage1.best_val_acc, finetuner_stage2.best_val_acc)
    
    finetuner_stage2.plot_history()
    finetuner_stage2.save_model('resnet50_layer4.pth')
    
    # Evaluasi di test set menggunakan model yang sudah dilatih
    print("\nEvaluating unfreeze-layer4 model on test set...")
    trained_model = finetuner_stage2.model  # model sudah dalam state terbaik
    evaluate_on_test(trained_model, test_loader)
    
    return finetuner_stage2