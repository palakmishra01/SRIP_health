from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# aggregated Accuracy
final_accuracy = accuracy_score(all_y_true, all_y_pred)
print(f'Final Aggregated Accuracy: {final_accuracy:.4f}')

# Mapping classes for clarity in the report
target_names = ['Normal (0)', 'Hypopnea (1)', 'Apnea (2)']
print('\nFinal Aggregated Classification Report:')
print(classification_report(all_y_true, all_y_pred, target_names=target_names))

#Display Confusion Matrix
conf_mat = confusion_matrix(all_y_true, all_y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Final Aggregated Confusion Matrix (LOPO Cross-Validation)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
