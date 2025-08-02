# pagmulat_api/train_model.py
import os
import pandas as pd
import joblib
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
try:
    from pagmulat_backend.pagmulat_api.utils.formatting import format_itemset
except ModuleNotFoundError:
    # Allow running as a script
    from utils.formatting import format_itemset

# Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def unified_training_pipeline(data_path, target_column, 
                             min_support=0.15, min_confidence=0.65, min_lift=1.2):
    """
    Unified training and ARM pipeline with frontend-aligned outputs
    """
    # Load and validate data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_excel(data_path)
    print(f"✅ Data loaded: {len(df)} records")
    
    # ========================================================================
    # 1. MODEL TRAINING
    # ========================================================================
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained: Accuracy = {accuracy:.2%}")
    
    # ========================================================================
    # 2. ASSOCIATION RULE MINING
    # ========================================================================
    # Convert to boolean for ARM
    df_bool = df.astype(bool)
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(
        df_bool, 
        min_support=min_support,
        use_colnames=True,
        low_memory=True
    )
    
    # Generate association rules
    rules = association_rules(
        frequent_itemsets, 
        metric="confidence", 
        min_threshold=min_confidence
    )
    filtered_rules = rules[rules['lift'] >= min_lift]
    print(f"✅ ARM completed: {len(filtered_rules)} rules generated")
    
    # ========================================================================
    # 3. SAVE ARTIFACTS WITH CONSISTENT FORMATTING
    # ========================================================================
    # Save model artifacts
    model_path = os.path.join(MODEL_DIR, 'student_behavior_model.pkl')
    encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    
    # Convert frozensets to strings for CSV
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(
        lambda x: ', '.join(sorted(x)) if x else ''
    )
    filtered_rules['antecedents'] = filtered_rules['antecedents'].apply(
        lambda x: ', '.join(sorted(x)) if x else ''
    )
    filtered_rules['consequents'] = filtered_rules['consequents'].apply(
        lambda x: ', '.join(sorted(x)) if x else ''
    )
    
    # Save ARM results
    itemsets_path = os.path.join(DATA_DIR, 'frequent_itemsets.csv')
    rules_path = os.path.join(DATA_DIR, 'association_rules_filtered.csv')
    frequent_itemsets.to_csv(itemsets_path, index=False)
    filtered_rules.to_csv(rules_path, index=False)
    
    # Prepare metadata
    metadata = {
        'model': {
            'accuracy': accuracy,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'target': target_column,
            'features': list(X.columns)
        },
        'arm': {
            'min_support': min_support,
            'min_confidence': min_confidence,
            'min_lift': min_lift,
            'num_rules': len(filtered_rules),
            'num_itemsets': len(frequent_itemsets),
            'top_rules': [
                {
                    'antecedent': format_itemset(row['antecedents']),
                    'consequent': format_itemset(row['consequents']),
                    'confidence': round(row['confidence'], 2)
                }
                for _, row in filtered_rules.sort_values(
                    'confidence', ascending=False).head(3).iterrows()
            ]
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(BASE_DIR, 'training_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Artifacts saved to:\n"
          f"- Models: {model_path}\n"
          f"- ARM Data: {itemsets_path}, {rules_path}\n"
          f"- Metadata: {metadata_path}")
    
    return metadata

if __name__ == "__main__":
    # Configuration - Match frontend requirements
    CONFIG = {
        'data_path': os.path.join(BASE_DIR, "ModifiedFinalData.xlsx"),
        'target_column': "Productive_Yes",
        'min_support': 0.15,
        'min_confidence': 0.65,
        'min_lift': 1.2
    }
    
    try:
        results = unified_training_pipeline(**CONFIG)
        print(f"🚀 Training complete! Model accuracy: {results['model']['accuracy']:.2%}")
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()