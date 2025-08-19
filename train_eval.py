import torch
import torch.nn.functional as F

def train_classifier(model, dataloader, criterion, optimizer, device='cpu'):
    model.train()
    model.to(device)
    total_loss = 0.0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_classifier(model, dataloader, device='cpu'):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def train_attacker(attack_model, shadow_model, train_loader, out_loader, optimizer, criterion, num_posteriors=3,
                   device='cpu'):
    attack_model.train()
    shadow_model.eval()
    attack_model.to(device)
    shadow_model.to(device)
    total_loss = 0.0
    for (train_images, _), (out_images, _) in zip(train_loader, out_loader):
        train_images, out_images = train_images.to(device), out_images.to(device)

        # Get posteriors from shadow model
        with torch.no_grad():
            train_posteriors = F.softmax(shadow_model(train_images), dim=1)
            out_posteriors = F.softmax(shadow_model(out_images), dim=1)

        # Sort and select top-k posteriors
        train_top_k, _ = torch.sort(train_posteriors, dim=1, descending=True)
        out_top_k, _ = torch.sort(out_posteriors, dim=1, descending=True)
        train_top_k = train_top_k[:, :num_posteriors]
        out_top_k = out_top_k[:, :num_posteriors]

        # Labels: 1 for train (member), 0 for out (non-member)
        train_labels = torch.ones(train_images.size(0), device=device)
        out_labels = torch.zeros(out_images.size(0), device=device)

        optimizer.zero_grad()
        train_preds = attack_model(train_top_k).squeeze()
        out_preds = attack_model(out_top_k).squeeze()

        loss = (criterion(train_preds, train_labels) + criterion(out_preds, out_labels)) / 2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_attacker(attack_model, target_model, train_loader, out_loader, num_posteriors=3, device='cpu'):
    attack_model.eval()
    target_model.eval()
    attack_model.to(device)
    target_model.to(device)
    thresholds = torch.arange(0.5, 0.81, 0.01).to(device)
    max_accuracy = 0.0

    with torch.no_grad():
        for (train_images, _), (out_images, _) in zip(train_loader, out_loader):
            train_images, out_images = train_images.to(device), out_images.to(device)

            # Get posteriors from target model
            train_posteriors = F.softmax(target_model(train_images), dim=1)
            out_posteriors = F.softmax(target_model(out_images), dim=1)

            # Sort and select top-k posteriors
            train_top_k, _ = torch.sort(train_posteriors, dim=1, descending=True)
            out_top_k, _ = torch.sort(out_posteriors, dim=1, descending=True)
            train_top_k = train_top_k[:, :num_posteriors]
            out_top_k = out_top_k[:, :num_posteriors]

            # Get predictions
            train_preds = attack_model(train_top_k).squeeze()
            out_preds = attack_model(out_top_k).squeeze()

            # Evaluate across thresholds
            for t in thresholds:
                correct = (train_preds >= t).sum().item() + (out_preds < t).sum().item()
                total = train_preds.size(0) + out_preds.size(0)
                accuracy = 100 * correct / total
                max_accuracy = max(max_accuracy, accuracy)

    return max_accuracy