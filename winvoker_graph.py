# import matplotlib.pyplot as plt

# # Final accuracy değerleri (senin çıktılarından alındı)
# final_acc = {
#     "Logistic Regression": 0.9344,
#     "SVM": 0.9330,
#     "MLP": 0.9421
# }

# # MLP'nin epoch bazlı (örnek) accuracy değerleri
# mlp_epochs = list(range(1, 6))
# mlp_acc = [0.72, 0.75, 0.78, 0.80, 0.82]

# # --- Grafik 1: MLP epoch bazlı accuracy ---
# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.plot(mlp_epochs, mlp_acc, marker="o", label="MLP")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("MLP Epoch Bazlı Accuracy")
# plt.legend()

# # --- Grafik 2: Final accuracy karşılaştırması ---
# plt.subplot(1, 2, 2)
# plt.bar(final_acc.keys(), final_acc.values(), color=["blue", "green", "orange"])
# plt.ylim(0.9, 1.0)
# plt.ylabel("Final Accuracy")
# plt.title("Modellerin Final Accuracy Karşılaştırması")

# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt

# # Senin MLP çıktından alınan training loss değerleri
# mlp_loss = [
#     0.21533132, 0.18226303, 0.17495849, 0.17021420, 0.16662175,
#     0.16386636, 0.16087957, 0.15801088, 0.15697573, 0.15428962,
#     0.15289219, 0.15110446, 0.15025591, 0.14850264, 0.14738550,
#     0.14568746, 0.14475097, 0.14416309, 0.14273669, 0.14178515,
#     0.14156339, 0.14076751, 0.13929293, 0.13882473, 0.13714543,
#     0.13743513, 0.13617066, 0.13517994, 0.13485866, 0.13414723,
#     0.13328153, 0.13275285, 0.13240406, 0.13195951, 0.13105079,
#     0.13056813, 0.12970454, 0.12948965, 0.12873818, 0.12795968,
#     0.12832780, 0.12702242, 0.12699430, 0.12658923, 0.12592588,
#     0.12528839, 0.12590995, 0.12509858, 0.12406149, 0.12351863
# ]

# epochs = range(1, len(mlp_loss) + 1)

# plt.plot(epochs, mlp_loss, marker="o", label="Training Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("MLP Training Loss")
# plt.legend()
# plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Confusion matrix hesapla
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_mlp = confusion_matrix(y_test, y_pred_mlp)

labels = ["Negative", "Notr", "Positive"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, cm, title in zip(axes, [cm_lr, cm_svm, cm_mlp], ["Logistic Regression", "SVM", "MLP"]):
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    # Her hücreye sayı yaz
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")

plt.tight_layout()
plt.show()
