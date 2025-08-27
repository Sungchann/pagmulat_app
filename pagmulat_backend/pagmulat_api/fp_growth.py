# pagmulat_api/train_model_fpgrowth.py
import os
import pandas as pd
from config import config
import json
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules

try:
    from .utils.formatting import format_itemset
except ImportError:
    # Allow running as a script
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils.formatting import format_itemset

# Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data/rules')
os.makedirs(DATA_DIR, exist_ok=True)

def fpgrowth_training_pipeline(data_path, min_support=0.1, min_confidence=0.4, min_lift=1.2):
    """
    FP-Growth-based Association Rule Mining pipeline
    Discovers interpretable rules and patterns for predictions/insights.
    This algorithm is generally faster and more memory efficient than Apriori.
    """
    # Load data
    df = pd.read_excel(data_path)
    
    # 1. FILTER NON-BINARY COLUMNS
    binary_cols = [col for col in df.columns if set(df[col].unique()).issubset({0, 1})]
    df = df[binary_cols]
    print(f"✅ Filtered to {len(df.columns)} binary features")
    
    # 2. FIND FREQUENT ITEMSETS USING FP-GROWTH
    # FP-Growth is a non-candidate-generation algorithm, which makes it faster.
    df_bool = df.astype(bool)
    frequent_itemsets = fpgrowth(
        df_bool, 
        min_support=min_support,
        use_colnames=True
    )
    print(f"✅ Found {len(frequent_itemsets)} frequent itemsets using FP-Growth")
    
    # Generate association rules from the frequent itemsets
    print(f"⛏️ Generating association rules (min_confidence={min_confidence}, min_lift={min_lift})...")
    rules = association_rules(
        frequent_itemsets, 
        metric="confidence", 
        min_threshold=min_confidence
    )
    
    # Filter by lift
    filtered_rules = rules[rules['lift'] >= min_lift]
    print(f"✅ Generated {len(filtered_rules)} raw association rules")
    
    # Sort by confidence for quality ranking
    filtered_rules = filtered_rules.sort_values(['confidence', 'lift'], ascending=[False, False])
    
    # LIMIT RULES FOR FRONTEND PERFORMANCE
    MAX_RULES_FOR_UI = 5000 
    if len(filtered_rules) > MAX_RULES_FOR_UI:
        filtered_rules_limited = filtered_rules.head(MAX_RULES_FOR_UI)
        print(f"🎯 Limited to top {MAX_RULES_FOR_UI} rules for UI (sorted by confidence & lift)")
    else:
        filtered_rules_limited = filtered_rules
        print(f"✅ Using all {len(filtered_rules)} rules (under limit)")
    
    # ========================================================================
    # PREDICTION CAPABILITY ANALYSIS
    # ========================================================================
    # Analyze rules that predict productivity outcomes (from limited set)
    productivity_rules = filtered_rules_limited[
        filtered_rules_limited['consequents'].astype(str).str.contains('Productive_', case=False, na=False)
    ]
    
    print(f"📊 Rules for productivity prediction: {len(productivity_rules)} (from top {len(filtered_rules_limited)})")
    
    # Show a sample of top and bottom prediction rules
    if len(productivity_rules) > 0:
        print("\n🎯 Top 5 Productivity Prediction Rules:")
        for i, (_, rule) in enumerate(productivity_rules.head(5).iterrows()):
            antecedent = ', '.join(sorted(rule['antecedents']))
            consequent = ', '.join(sorted(rule['consequents']))
            print(f"  {i+1}. IF [{antecedent}] THEN [{consequent}] "
                  f"(Confidence: {rule['confidence']:.1%}, Lift: {rule['lift']:.2f})")
        
        print("\n🎯 Bottom 5 Productivity Prediction Rules:")
        for i, (_, rule) in enumerate(productivity_rules.tail(5).iterrows()):
            antecedent = ', '.join(sorted(rule['antecedents']))
            consequent = ', '.join(sorted(rule['consequents']))
            print(f"  {i+1}. IF [{antecedent}] THEN [{consequent}] "
                  f"(Confidence: {rule['confidence']:.1%}, Lift: {rule['lift']:.2f})")
    
    # ========================================================================
    # SAVE ARTIFACTS
    # ========================================================================
    # Convert frozensets to strings for CSV storage
    frequent_itemsets_export = frequent_itemsets.copy()
    frequent_itemsets_export['itemsets'] = frequent_itemsets_export['itemsets'].apply(
        lambda x: ', '.join(sorted(x)) if x else ''
    )
    
    # Use the limited rules for UI export
    rules_export = filtered_rules_limited.copy()
    rules_export['antecedents'] = rules_export['antecedents'].apply(
        lambda x: ', '.join(sorted(x)) if x else ''
    )
    rules_export['consequents'] = rules_export['consequents'].apply(
        lambda x: ', '. join(sorted(x)) if x else ''
    )
    
    # NEW: Print a sample of the confidence column to the console for debugging
    print("\n🔬 Debugging Confidence Values Before Saving:")
    print("Top 5 confidence values:\n", rules_export['confidence'].head(5))
    print("Bottom 5 confidence values:\n", rules_export['confidence'].tail(5))
    
    # Save ARM results (limited set for UI performance)
    itemsets_path = os.path.join(DATA_DIR, 'fpgrowth_frequent_itemsets.csv')
    rules_path = os.path.join(DATA_DIR, 'fpgrowth_association_rules.csv')
    frequent_itemsets_export.to_csv(itemsets_path, index=False)
    rules_export.to_csv(rules_path, index=False)
    
    # Prepare metadata for frontend
    metadata = {
        'model': {
            'algorithm': 'FP-Growth Association Rule Mining',
            'data_records': len(df),
            'features': list(df.columns),
            'frequent_itemsets': len(frequent_itemsets),
            'total_rules': len(filtered_rules), 
            'total_rules_ui': len(filtered_rules_limited), 
            'productivity_rules': len(productivity_rules),
            'top_confidence': float(filtered_rules_limited['confidence'].max()) if len(filtered_rules_limited) > 0 else 0,
            'avg_confidence': float(filtered_rules_limited['confidence'].mean()) if len(filtered_rules_limited) > 0 else 0
        },
        'arm': {
            'min_support': min_support,
            'min_confidence': min_confidence,
            'min_lift': min_lift,
            'num_rules': len(filtered_rules_limited), 
            'num_itemsets': len(frequent_itemsets),
            'top_rules': [
                {
                    'antecedent': format_itemset(row['antecedents']),
                    'consequent': format_itemset(row['consequents']),
                    'confidence': round(row['confidence'], 3),
                    'lift': round(row['lift'], 2),
                    'support': round(row['support'], 3)
                }
                for _, row in filtered_rules_limited.head(5).iterrows()
            ]
        }
    }
    
    # Save metadata
    metadata_path = os.path.join(BASE_DIR, 'training_metadata_fpgrowth.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Artifacts saved:")
    print(f"  • Frequent itemsets: {itemsets_path}")
    print(f"  • Association rules: {rules_path}")
    print(f"  • Metadata: {metadata_path}")
    
    return metadata

if __name__ == "__main__":
    CONFIG = {
        'data_path': os.path.join(BASE_DIR, "ModifiedFinalData.xlsx"),
        'min_support': config.MIN_SUPPORT, 
        'min_confidence': config.MIN_CONFIDENCE, 
        'min_lift': config.MIN_LIFT
    }
    
    try:
        results = fpgrowth_training_pipeline(**CONFIG)
        print(f"\nFP-Growth Training Complete!")
        print(f"  • Total Rules Generated: {results['arm']['num_rules']}")
        print(f"  • Productivity Rules: {results['model']['productivity_rules']}")
        print(f"  • Average Confidence: {results['model']['avg_confidence']:.1%}")
        print(f"  • Top Confidence: {results['model']['top_confidence']:.1%}")
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
