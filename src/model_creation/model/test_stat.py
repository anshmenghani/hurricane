import pandas as pd
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def evaluate_fold_mae(fold_num):
    true_df = pd.read_csv(f"fold_{fold_num}.csv")[["lat", "long"]].dropna().reset_index(drop=True)
    pred_df = pd.read_csv(f"predictions_fold{fold_num}.csv")

    min_len = min(len(true_df), len(pred_df))
    true = true_df.iloc[:min_len]
    pred = pred_df.iloc[:min_len]

    mae_lat = mean_absolute_error(true["lat"], pred["pred_lat"])
    mae_long = mean_absolute_error(true["long"], pred["pred_long"])

    return mae_lat, mae_long

def kf():
    acc_list = []
    for fold in range(1, 6):
        mlat, mlong = evaluate_fold_mae(fold)
        acc_list.append((mlat + mlong) / 2)

    return [100 - acc_list[i] for i in range(len(acc_list))]
    

def plot_k_fold(accuracies):
    mean_acc = sum(accuracies) / len(accuracies)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 6), accuracies, marker='o', linestyle='-', label='Accuracy per Fold')
    plt.axhline(y=mean_acc, color='r', linestyle='--', label=f'Mean Accuracy: {mean_acc:.2f}%')

    plt.xticks(range(1, 6))
    plt.xlabel("Fold")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy per Fold with Mean Line")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def recall_presc_f1():
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_folds = 5

    for fold in range(1, num_folds + 1):
        y_true_df = pd.read_csv(f"fold_{fold}.csv")[["lat", "long"]].reset_index(drop=True)
        y_pred_df = pd.read_csv(f"predictions_fold{fold}.csv")[["pred_lat", "pred_long"]].reset_index(drop=True)

        min_len = min(len(y_true_df), len(y_pred_df))
        y_true_df = y_true_df.iloc[:min_len]
        y_pred_df = y_pred_df.iloc[:min_len]

        y_true_labels = y_true_df.round(1).astype(str).agg('_'.join, axis=1)
        y_pred_labels = y_pred_df.round(1).astype(str).agg('_'.join, axis=1)

        total_precision += precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
        total_recall += recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
        total_f1 += f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)

    avg_precision = total_precision / num_folds
    avg_recall = total_recall / num_folds
    avg_f1 = total_f1 / num_folds
    return avg_precision, avg_recall, avg_f1


if __name__ == "__main__":
    accs = kf()
    plot_k_fold(accs)
    stats = recall_presc_f1()
    print(stats)
