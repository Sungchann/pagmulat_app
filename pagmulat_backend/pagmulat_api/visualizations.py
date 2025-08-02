import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

def behavior_distribution_plot():
    """Generate behavior distribution plot"""
    df = pd.read_excel("ModifiedFinalData.xlsx")
    
    # Example: Burnout distribution
    fig, ax = plt.subplots()
    df['Burnout_Exhaustion'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Burnout Frequency Distribution')
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')