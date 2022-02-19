import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_predict):
    cm = confusion_matrix(y_true, y_predict)
    sns.heatmap(cm, cmap='Blues', annot=True, fmt='d', cbar=False, xticklabels=['Non_Cancer_Prediction', 'Cancer_Prediction'],
                yticklabels=['Non_Cancer', 'Cancer'], linewidths='0.1', linecolor='black')
    plt.show()
