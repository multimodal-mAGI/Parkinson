import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

def train_model(model, train_loader, test_loader, device, save_path, epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    best_auc = 0.0

    for epoch in range(epochs):
        model.train()
        train_losses, train_accs = [], []
        for audio_batch, img_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            audio_batch, img_batch, y_batch = audio_batch.to(device), img_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(audio_batch, img_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            acc = accuracy_score(y_batch.cpu(), preds.cpu())
            train_losses.append(loss.item())
            train_accs.append(acc)

        # Validation
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for audio_batch, img_batch, y_batch in test_loader:
                audio_batch, img_batch = audio_batch.to(device), img_batch.to(device)
                logits = model(audio_batch, img_batch)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y_batch.numpy())

        auc = roc_auc_score(all_labels, all_probs)
        print(f"Epoch {epoch+1} | Loss {np.mean(train_losses):.4f} | Acc {np.mean(train_accs):.4f} | ValAUC {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved (AUC={best_auc:.4f})")

    return best_auc
