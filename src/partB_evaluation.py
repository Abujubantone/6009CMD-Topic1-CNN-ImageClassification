# 6009CMD Topic 1 - Part B Evaluation (GenAI assisted) - MEMORY SAFE
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("=== Part B Evaluation - MC Dropout + Entropy (light & safe) ===")

model = tf.keras.models.load_model("PartB/models/partB_best_model.h5")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype("float32") / 255.0
y_test = y_test.flatten()
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Safe MC Dropout (5 iterations + batches of 1000)
def mc_dropout_predict(model, x_batch, n_iter=5):
    preds = [model(x_batch, training=True).numpy() for _ in range(n_iter)]
    return np.mean(preds, axis=0)

batch_size = 1000
probs_list = []
for i in range(0, len(x_test), batch_size):
    batch = x_test[i:i+batch_size]
    mean_p = mc_dropout_predict(model, batch)
    probs_list.append(mean_p)

probs = np.concatenate(probs_list, axis=0)
pred_class = np.argmax(probs, axis=1)
entropy_scores = -np.sum(probs * np.log(probs + 1e-10), axis=-1)

print(f"Test Accuracy: {np.mean(pred_class == y_test):.4f}")
print(f"Ambiguous images (entropy > 1.5): {np.sum(entropy_scores > 1.5)}")

# Confusion Matrix
cm = confusion_matrix(y_test, pred_class)
plt.figure(figsize=(10,8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("Part B: Confusion Matrix (GenAI-improved)")
plt.savefig("PartB/reports/figures/partB_confusion_matrix.png")
plt.show()

# Top 9 ambiguous (medical X-ray style)
top_idx = np.argsort(entropy_scores)[-9:]
plt.figure(figsize=(12,12))
for i, idx in enumerate(top_idx):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[idx])
    plt.title(f"Case {i+1}: Entropy={entropy_scores[idx]:.2f}\nTrue: {class_names[y_test[idx]]}\nPred: {class_names[pred_class[idx]]}")
    plt.axis("off")
plt.suptitle("Part B: Top Ambiguous Cases (MC Dropout entropy)")
plt.savefig("PartB/reports/figures/partB_ambiguous_samples.png")
plt.show()

print("✅ Part B Evaluation COMPLETE! (safe version)")
print("Files saved in PartB/reports/figures/")
