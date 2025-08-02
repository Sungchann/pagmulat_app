import pandas as pd
from django.conf import settings
from mlxtend.frequent_patterns import apriori, association_rules

def load_arm_data():
    from .models import AssociationRule  # Import moved inside function
    """Loads ARM rules into database on app startup"""
    # Skip if rules already exist
    if AssociationRule.objects.exists():
        return
    
    try:
        # Load data from project root
        data_path = settings.BASE_DIR / 'ModifiedFinalData.xlsx'
        df = pd.read_excel(data_path)
        df_bool = df.astype(bool)
        
        # Generate association rules
        frequent_itemsets = apriori(df_bool, min_support=0.1, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
        filtered_rules = rules[rules['lift'] >= 1.2]
        
        # Create model instances
        rules_to_create = []
        for _, row in filtered_rules.iterrows():
            rules_to_create.append(AssociationRule(
                antecedents=list(row['antecedents']),
                consequents=list(row['consequents']),
                support=row['support'],
                confidence=row['confidence'],
                lift=row['lift']
            ))
        
        # Bulk create for efficiency
        AssociationRule.objects.bulk_create(rules_to_create)
        print(f"Loaded {len(rules_to_create)} ARM rules into database")
    
    except Exception as e:
        print(f"Error loading ARM data: {str(e)}")