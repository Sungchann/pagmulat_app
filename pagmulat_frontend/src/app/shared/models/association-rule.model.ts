export interface AssociationRule {
  antecedent: string;
  consequent: string;
  confidence: number;
  lift: number;
  support: number;
}

export interface ArmMetadata {
  model: {
    algorithm: string;
    data_records: number;
    features: string[];
    frequent_itemsets: number;
    total_rules: number;
    productivity_rules: number;
    top_confidence: number;
    avg_confidence: number;
  };
  arm: {
    min_support: number;
    min_confidence: number;
    min_lift: number;
    num_rules: number;
    num_itemsets: number;
    top_rules: AssociationRule[];
  };
}
