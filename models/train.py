import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, test_loader, lr, weight_decay, epochs, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device
        
        # Calculer les poids des classes
        labels = [label for _, label in train_loader.dataset]
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def train(self, save_path='model.pth'):
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch") as t:
                for images, labels in t:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    t.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = 100 * correct / total
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_accuracy)

            # Évaluation sur le test set
            test_acc = self.evaluate_epoch()
            self.test_accuracies.append(test_acc)

            # Ajuster le taux d'apprentissage
            self.scheduler.step(test_acc)

            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, Test Accuracy: {test_acc:.2f}%")

        # Sauvegarder le modèle
        torch.save(self.model.state_dict(), save_path)
        print(f"Modèle sauvegardé sous {save_path}")

        # Créer et sauvegarder le graphique combiné
        epochs = range(1, self.epochs + 1)
        fig, ax1 = plt.subplots(figsize=(8, 5))

        color_loss = 'tab:blue'
        ax1.set_xlabel('Époque')
        ax1.set_ylabel('Perte', color=color_loss)
        ax1.plot(epochs, self.train_losses, color=color_loss, label='Perte')
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax2 = ax1.twinx()  # Instancier un second axe Y qui partage le même axe X
        color_acc = 'tab:red'
        ax2.set_ylabel('Accuracy (%)', color=color_acc)
        ax2.plot(epochs, self.train_accuracies, color=color_acc, label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)

        plt.title('Training Loss and Accuracy (PyTorch)')
        fig.tight_layout()  # Pour éviter les chevauchements
        plt.savefig('training_history_pytorch.png')
        plt.close()

    def evaluate_epoch(self):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        return 100.0 * total_correct / total_samples

    def evaluate(self):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        accuracy = 100.0 * total_correct / total_samples
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy