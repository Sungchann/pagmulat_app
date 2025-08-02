# pagmulat_api/utils/formatting.py
ITEM_NAME_MAP = {
    'Year_1': 'Year 1',
    'Year_2': 'Year 2',
    'Year_3': 'Year 3',
    'Year_4': 'Year 4',
    'ChatGPT_AI_Usage_Always': 'Uses ChatGPT Regularly',
    'Social_Media_Distraction_Often': 'High Social Media Distraction',
    'ProductivityTool_Notion': 'Uses Notion',
    'Fixed_Study_Schedule_Yes': 'Structured Study',
    'Social_Distraction_Rarely': 'Low Procrastination',
    'Code_Compiler_Usage_Often': 'Uses GitHub',
    'Online_Collaboration_Tools_Always': 'Uses Collaboration Tools',
    'Burnout_Exhaustion_Often': 'Frequent Burnout',
    'Study_Start_Time_Late Night (after 10pm)': 'Late Night Studying',
    'Productive_Yes': 'Highly Productive',
    'Productive_No': 'Low Productivity',
    'Productive_Sometimes': 'Moderate Productivity',
    # Add all other mappings here
}

def format_itemset(itemset):
    """
    Convert raw itemset to human-readable format
    Example: 
        Input: "frozenset({'Year_3', 'ChatGPT_AI_Usage_Always'})"
        Output: "Year 3, Uses ChatGPT Regularly"
    """
    if isinstance(itemset, str):
        # Clean and parse string representation
        items = itemset.replace('frozenset({', '') \
                       .replace('})', '') \
                       .replace("'", "") \
                       .split(', ')
    elif isinstance(itemset, (frozenset, list, set)):
        # Handle direct collection types
        items = sorted(list(itemset))
    else:
        return str(itemset)
    
    # Map to human-readable names
    return ', '.join([ITEM_NAME_MAP.get(item.strip(), item.replace('_', ' ')) 
                      for item in items if item.strip()])