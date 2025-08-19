import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataloader import get_loaders
from model import ShadowModel, AttackModel
from train_eval import train_classifier, evaluate_classifier, train_attacker, evaluate_attacker


def main():
    # Configuration
    batch_size = 64
    shadow_epochs = 20
    target_epochs = 20
    attack_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)

    # Data loaders
    shadow_train_loader = get_loaders(batch_size_train=batch_size, split="shadow_train")
    shadow_out_loader = get_loaders(batch_size_train=batch_size, split="shadow_out")
    target_train_loader = get_loaders(batch_size_train=batch_size, split="target_train")
    target_out_loader = get_loaders(batch_size_train=batch_size, split="target_out")
    test_loader = get_loaders(batch_size_train=batch_size, split="test")

    # Initialize models
    shadow_model = ShadowModel().to(device)
    target_model = ShadowModel().to(device)
    attack_model = AttackModel(input_size=3).to(device)

    # Loss and optimizers
    classifier_criterion = nn.CrossEntropyLoss()
    attack_criterion = nn.BCELoss()
    shadow_optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)
    target_optimizer = optim.Adam(target_model.parameters(), lr=0.001)
    attack_optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

    # Train shadow model
    print("Training Shadow Model...")
    for epoch in range(shadow_epochs):
        loss = train_classifier(shadow_model, shadow_train_loader, classifier_criterion, shadow_optimizer, device)
        if (epoch + 1) % 5 == 0:
            train_acc = evaluate_classifier(shadow_model, shadow_train_loader, device)
            test_acc = evaluate_classifier(shadow_model, test_loader, device)
            print(
                f"Epoch {epoch + 1}/{shadow_epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    torch.save(shadow_model.state_dict(), os.path.join(model_dir, "shadow_cifar10.pth"))

    # Train target model
    print("\nTraining Target Model...")
    for epoch in range(target_epochs):
        loss = train_classifier(target_model, target_train_loader, classifier_criterion, target_optimizer, device)
        if (epoch + 1) % 5 == 0:
            train_acc = evaluate_classifier(target_model, target_train_loader, device)
            test_acc = evaluate_classifier(target_model, test_loader, device)
            print(
                f"Epoch {epoch + 1}/{target_epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    torch.save(target_model.state_dict(), os.path.join(model_dir, "target_cifar10.pth"))

    # Train attack model
    print("\nTraining Attack Model...")
    for epoch in range(attack_epochs):
        loss = train_attacker(attack_model, shadow_model, shadow_train_loader, shadow_out_loader, attack_optimizer,
                              attack_criterion, device=device)
        if (epoch + 1) % 5 == 0:
            attack_acc = evaluate_attacker(attack_model, target_model, target_train_loader, target_out_loader,
                                           device=device)
            print(f"Epoch {epoch + 1}/{attack_epochs}, Loss: {loss:.4f}, Attack Acc: {attack_acc:.2f}%")
    torch.save(attack_model.state_dict(), os.path.join(model_dir, "attack_cifar10.pth"))

    # Final evaluation
    print("\nFinal Evaluation:")
    shadow_acc = evaluate_classifier(shadow_model, test_loader, device)
    target_acc = evaluate_classifier(target_model, test_loader, device)
    attack_acc = evaluate_attacker(attack_model, target_model, target_train_loader, target_out_loader, device=device)
    print(f"Shadow Model Test Accuracy: {shadow_acc:.2f}%")
    print(f"Target Model Test Accuracy: {target_acc:.2f}%")
    print(f"Attack Model Accuracy: {attack_acc:.2f}%")


if __name__ == "__main__":
    main()