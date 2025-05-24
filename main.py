import argparse
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from models.cnn import CustomCNN
from models.cnn_tf import create_cnn_model
from models.train import Trainer
from utils.prep import get_pytorch_dataloaders, get_tensorflow_generators
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate brain tumor CNN models")
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.0005)  # Réduit pour un apprentissage plus fin
    parser.add_argument('--wd', type=float, default=1e-3)   # Augmenté pour plus de régularisation
    parser.add_argument('--cuda', action='store_true')
    return parser.parse_args()

def train_pytorch_model(args, device):
    print("Training PyTorch model...")
    train_loader, test_loader, classes = get_pytorch_dataloaders(data_dir='dataset')
    model = CustomCNN(num_classes=len(classes)).to(device)
    trainer = Trainer(model, train_loader, test_loader, args.lr, args.wd, args.epochs, device)
    trainer.train(save_path='model.pth')
    trainer.evaluate()
    print("PyTorch model saved at model.pth")

def train_tensorflow_model(args):
    print("Training TensorFlow model...")
    train_generator, test_generator, class_indices = get_tensorflow_generators(data_dir='dataset')
    model = create_cnn_model(num_classes=len(class_indices))
    
    # Calculer les poids des classes
    labels = train_generator.classes
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = dict(enumerate(class_weights))
    
    # Ajouter un planificateur de taux d'apprentissage
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=args.epochs,
        verbose=1,
        class_weight=class_weights_dict,
        callbacks=[reduce_lr]
    )
    loss, accuracy = model.evaluate(test_generator, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}% | Test Loss: {loss:.4f}")
    model.save('model_tf.h5')
    print("TensorFlow model saved at model_tf.h5")

    # Créer et sauvegarder le graphique combiné pour TensorFlow
    epochs = range(1, args.epochs + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_loss = 'tab:blue'
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Perte', color=color_loss)
    ax1.plot(epochs, history.history['loss'], color=color_loss, label='Perte')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    ax2 = ax1.twinx()  # Instancier un second axe Y qui partage le même axe X
    color_acc = 'tab:red'
    ax2.set_ylabel('Accuracy (%)', color=color_acc)
    ax2.plot(epochs, [acc * 100 for acc in history.history['accuracy']], color=color_acc, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    plt.title('Training Loss and Accuracy (TensorFlow)')
    fig.tight_layout()  # Pour éviter les chevauchements
    plt.savefig('training_history_tf.png')
    plt.close()

def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == 'train':
        #train_pytorch_model(args, device)
        train_tensorflow_model(args)
    elif args.mode == 'eval':
        train_loader, test_loader, classes = get_pytorch_dataloaders(data_dir='dataset')
        model = CustomCNN(num_classes=len(classes)).to(device)
        try:
            model.load_state_dict(torch.load('model.pth', map_location=device))
            print("Loaded model.pth")
        except FileNotFoundError:
            print("Error: model.pth not found")
            return
        trainer = Trainer(model, train_loader, test_loader, args.lr, args.wd, args.epochs, device)
        trainer.evaluate()

if __name__ == '__main__':
    main()