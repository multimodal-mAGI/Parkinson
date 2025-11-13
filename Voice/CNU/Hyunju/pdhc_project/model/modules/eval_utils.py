import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, accuracy_score, roc_curve
)

def evaluate_model(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for audio_batch, img_batch, y_batch in loader:
            audio_batch, img_batch = audio_batch.to(device), img_batch.to(device)
            logits = model(audio_batch, img_batch)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    preds = (np.array(all_probs) > 0.5).astype(int)
    auc_val = roc_auc_score(all_labels, all_probs)
    acc_val = accuracy_score(all_labels, preds)
    print(f"\nAUC: {auc_val:.4f} | Accuracy: {acc_val:.4f}")
    print(classification_report(all_labels, preds, target_names=["Healthy", "Parkinson"], digits=4))

    # === 1 Figure에 2개의 Subplot 배치 ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Parkinson"])
    disp.plot(cmap="Blues", ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix")

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_title(f"ROC Curve (AUC={auc_val:.4f})")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")

    plt.tight_layout()
    plt.show()
