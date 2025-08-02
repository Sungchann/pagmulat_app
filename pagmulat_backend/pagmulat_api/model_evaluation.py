import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import json

def test_model(model_path, features_path, labels_path):
    """Evaluates model performance and generates visual reports"""
    # Load artifacts
    model = joblib.load(model_path)
    X_test = pd.read_csv(features_path)
    y_test = pd.read_csv(labels_path).squeeze()
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Metrics
    report = {
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'roc_auc': {}
    }
    
    # ROC Curve for each class (multiclass)
    plt.figure(figsize=(10, 8))
    for i in range(y_proba.shape[1]):
        fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        report['roc_auc'][f'Class_{i}'] = roc_auc
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig('roc_curves.png')
    plt.close()
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature Importance
    importance = pd.read_csv('../../data/feature_importance.csv', names=['feature', 'importance'])
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance.head(20))
    plt.title('Top 20 Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return report

if __name__ == "__main__":
    # Load previous results
    with open('training_results.json') as f:
        training_results = json.load(f)
    
    # Run testing
    test_results = test_model(
        model_path='../../models/student_behavior_model.pkl',
        features_path='../../data/test_features.csv',
        labels_path='../../data/test_labels.csv'
    )
    
    # Combine and save results
    full_report = {**training_results, **test_results}
    with open('full_test_report.json', 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print("Testing completed!")
    print("Visualizations saved: roc_curves.png, confusion_matrix.png, feature_importance.png")
