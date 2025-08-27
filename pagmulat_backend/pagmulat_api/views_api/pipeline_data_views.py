import pandas as pd
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
import os

def get_file_content(file_path):
    """
    Reads a file (CSV or Excel) and returns its headers and all its rows.
    
    Args:
        file_path (str): The full path to the file.
        
    Returns:
        dict: A dictionary containing 'headers' (list of column names) and 
              'rows' (list of dictionaries for all rows).
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Only CSV and Excel are supported.")

        file_content = {
            'headers': list(df.columns),
            'rows': df.to_dict(orient='records')
        }
        return file_content

    except Exception as e:
        raise Exception(f"Failed to read file '{file_path}': {str(e)}")


@api_view(['GET'])
def get_raw_data(request):
    """
    API endpoint to get the raw student data.
    Reads the full CSV, replaces NaN values with None, then returns headers and all rows.
    """
    file_path = os.path.join(settings.BASE_DIR, 'data/raw/', 'student_raw.csv')
    try:
        # Read the full raw CSV file
        df = pd.read_csv(file_path)
        
        # Retain the specific cleaning step: replace NaN values with None
        df = df.replace({np.nan: None})
        
        # Create the data dictionary with all rows from the cleaned DataFrame
        data = {
            'headers': list(df.columns),
            'rows': df.to_dict(orient='records')
        }
        
        return Response(data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def get_cleaned_data(request):
    """
    API endpoint to get the cleaned student data.
    Returns headers and all rows using the helper function.
    """
    file_path = os.path.join(settings.BASE_DIR, 'data/transformed/', 'transformed_student_data.xlsx')
    try:
        # Use the helper function to get all data
        data = get_file_content(file_path)
        return Response(data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def get_augmented_data(request):
    """
    API endpoint to get the augmented student data.
    Returns headers and all rows using the helper function.
    """
    file_path = os.path.join(settings.BASE_DIR, 'data/augmented/', 'augmented_student_data.xlsx')
    try:
        # Use the helper function to get all data
        data = get_file_content(file_path)
        return Response(data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)
