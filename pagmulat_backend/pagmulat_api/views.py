# pagmulat_api/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
from django.core.cache import cache
import pandas as pd
import os
import json
import joblib
import numpy as np
from .utils.formatting import format_itemset

# Helper function to load association rules
def load_association_rules():
    rules_path = os.path.join(settings.BASE_DIR, 'data', 'association_rules_filtered.csv')
    rules = pd.read_csv(rules_path)
    return rules.replace([np.nan, np.inf, -np.inf], None)

@api_view(['GET'])
def arm_dashboard(request):
    """Dashboard endpoint for ARM results"""
    # Check cache first
    cache_key = 'arm_dashboard_data'
    cached_data = cache.get(cache_key)
    if cached_data:
        return Response(cached_data)
    
    try:
        # Load metadata
        metadata_path = os.path.join(settings.BASE_DIR, 'training_metadata.json')
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Load ARM rules
        rules = load_association_rules()
        
        # Get top 5 rules by confidence
        top_rules = rules.sort_values('confidence', ascending=False).head(5)
        
        # Format rules for frontend
        formatted_rules = []
        for _, row in top_rules.iterrows():
            formatted_rules.append({
                'antecedent': format_itemset(row['antecedents']),
                'consequent': format_itemset(row['consequents']),
                'support': round(row['support'], 2),
                'confidence': round(row['confidence'], 2),
                'lift': round(row['lift'], 2)
            })
        
        # Prepare metrics
        active_students = metadata['model']['train_size'] + metadata['model']['test_size']
        
        # Build response
        response_data = {
            'metrics': {
                'active_students': active_students,
                'arm_rules': metadata['arm']['num_rules'],
                'frequent_itemsets': metadata['arm']['num_itemsets'],
                'model_accuracy': round(metadata['model']['accuracy'], 4)
            },
            'thresholds': {
                'support': {
                    'value': metadata['arm']['min_support'],
                    'display': f"≥ {metadata['arm']['min_support']}"
                },
                'confidence': {
                    'value': metadata['arm']['min_confidence'],
                    'display': f"≥ {metadata['arm']['min_confidence']}"
                },
                'lift': {
                    'value': metadata['arm']['min_lift'],
                    'display': f"> {metadata['arm']['min_lift']}"
                }
            },
            'rules_table': formatted_rules
        }
        
        # Cache for 1 hour
        cache.set(cache_key, response_data, timeout=3600)
        return Response(response_data)
        
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
def behavior_patterns(request, behavior):
    """Find rules related to specific behavior"""
    try:
        rules = load_association_rules()
        
        # Case-insensitive search
        behavior_lower = behavior.lower()
        filtered = rules[
            rules['antecedents'].str.lower().str.contains(behavior_lower) |
            rules['consequents'].str.lower().str.contains(behavior_lower)
        ]
        
        # Format results
        formatted = []
        for _, row in filtered.sort_values('confidence', ascending=False).head(20).iterrows():
            formatted.append({
                'pattern': format_itemset(row['antecedents']),
                'consequence': format_itemset(row['consequents']),
                'confidence': round(row['confidence'], 2),
                'support': round(row['support'], 2),
                'lift': round(row['lift'], 2)
            })
        
        return Response({
            'behavior': behavior,
            'patterns': formatted,
            'count': len(formatted)
        })
        
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def predict(request):
    """Make productivity predictions"""
    try:
        print("PREDICT INPUT:", request.data)  # Debug: print incoming data
        # Load model and encoder
        model_path = os.path.join(settings.BASE_DIR, 'models', 'student_behavior_model.pkl')
        encoder_path = os.path.join(settings.BASE_DIR, 'models', 'label_encoder.pkl')
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        # Load feature list from metadata
        metadata_path = os.path.join(settings.BASE_DIR, 'training_metadata.json')
        with open(metadata_path) as f:
            metadata = json.load(f)
            feature_names = metadata['model']['features']
        # Build one-hot encoded feature vector from survey input
        features_dict = {f: 0 for f in feature_names}
        # Map survey fields to one-hot features
        data = request.data
        # Example mapping (customize as needed for your survey fields)
        # Year
        if 'year_level' in data:
            features_dict[f"Year_{data['year_level'][-1]}"] = 1
        # Program
        if 'program' in data:
            prog = data['program'].split()[-1]  # e.g., 'CS', 'IT', 'EMC'
            features_dict[f"Program_{prog}"] = 1
        # Age
        if 'age' in data:
            age = int(data['age'])
            if age < 18:
                features_dict['Age_Under18'] = 1
            elif 18 <= age <= 20:
                features_dict['Age_18_20'] = 1
            elif 21 <= age <= 23:
                features_dict['Age_21_23'] = 1
            elif 24 <= age <= 26:
                features_dict['Age_24_26'] = 1
            elif 27 <= age <= 29:
                features_dict['Age_27_29'] = 1
            else:
                features_dict['Age_30_Plus'] = 1
        # LMS Usage
        if 'lms_usage' in data:
            features_dict[f"LMS_Usage_{data['lms_usage']}"] = 1
        # Code Platform Usage
        if 'code_platform_usage' in data:
            features_dict[f"Code_Compiler_Usage_{data['code_platform_usage']}"] = 1
        # AI Usage
        if 'ai_usage' in data:
            features_dict[f"ChatGPT_AI_Usage_{data['ai_usage']}"] = 1
        # Social Media Usage
        if 'social_media_usage' in data:
            features_dict[f"Social_Media_Daily_Visit_{data['social_media_usage']}"] = 1
        # Fixed Study Schedule
        if 'fixed_study_schedule' in data:
            features_dict[f"Fixed_Study_Schedule_{data['fixed_study_schedule']}"] = 1
        # Study Hours
        if 'study_hours' in data:
            features_dict[f"Study_Hours_Outside_Class_{data['study_hours']}"] = 1
        # Study Start Time
        if 'study_start_time' in data:
            features_dict[f"Study_Start_Time_{data['study_start_time']}"] = 1
        # Assignment Timeliness
        if 'assignment_timeliness' in data:
            features_dict[f"Submit_Assignments_On_Time_{data['assignment_timeliness']}"] = 1
        # Collaboration Tools Usage
        if 'collab_tools_usage' in data:
            features_dict[f"Online_Collaboration_Tools_{data['collab_tools_usage']}"] = 1
        # Sleep Time
        if 'sleep_time' in data:
            features_dict[f"Sleep_Time_Weekdays_{data['sleep_time']}"] = 1
        # Burnout
        if 'burnout' in data:
            features_dict[f"Burnout_Exhaustion_{data['burnout']}"] = 1
        # Study Breaks
        if 'study_breaks' in data:
            features_dict[f"Regular_Breaks_Studying_{data['study_breaks']}"] = 1
        # Motivation
        if 'motivation' in data:
            features_dict[f"Motivation_{data['motivation']}"] = 1
        # Motivation Triggers (multi-select)
        if 'motivation_triggers' in data:
            for trig in data['motivation_triggers']:
                features_dict[f"MotivationTrigger_{trig}"] = 1
        # Productivity
        if 'productivity' in data:
            features_dict[f"Productive_{data['productivity']}"] = 1
        # Productivity Tools (multi-select)
        if 'productivity_tools' in data:
            for tool in data['productivity_tools']:
                features_dict[f"ProductivityTool_{tool}"] = 1
        # Social Media Distraction
        if 'social_media_distraction' in data:
            features_dict[f"Social_Distraction_{data['social_media_distraction']}"] = 1
        # Distraction Platform
        if 'distraction_platform' in data:
            features_dict[f"Distracting_Platform_{data['distraction_platform']}"] = 1
        # Block Distractions
        if 'block_distractions' in data:
            features_dict[f"Distraction_Blocking_{data['block_distractions']}"] = 1
        # Academic Tools (multi-select)
        if 'academic_tools' in data:
            for tool in data['academic_tools']:
                features_dict[f"AcademicPlatform_{tool}"] = 1
        # Digital Reliance
        if 'digital_reliance' in data:
            features_dict[f"Over_Reliance_{data['digital_reliance']}"] = 1
        # Digital Habits/Change/Reflection (free text) are ignored for model
        # Build DataFrame for prediction
        X = pd.DataFrame([features_dict[f] for f in feature_names]).T
        X.columns = feature_names
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        # Map integer prediction to string label using metadata if needed
        # Try to get class mapping from metadata, fallback to encoder.classes_
        class_map = None
        if 'class_map' in metadata['model']:
            class_map = metadata['model']['class_map']
        if class_map:
            # class_map keys may be strings, ensure correct type
            class_label = class_map.get(str(prediction), class_map.get(int(prediction), prediction))
        else:
            # fallback: try encoder.classes_ (may be int or str)
            class_label = encoder.classes_[prediction]
        print(f"[DEBUG] class_label value: {class_label} (type: {type(class_label)})")
        readable_labels = {
            'Productive_Yes': 'Highly Productive',
            'Productive_No': 'Low Productivity',
            'Productive_Sometimes': 'Moderate Productivity'
        }
        # Ensure class_label is a string for mapping
        class_label_str = str(class_label)
        if class_label_str not in readable_labels:
            print(f"[WARNING] class_label '{class_label_str}' not found in readable_labels. Returning raw label.")
        from datetime import datetime
        history_path = os.path.join(settings.BASE_DIR, 'prediction_history.json')
        # Convert np types to native Python types for JSON
        def to_py(val):
            if isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            if isinstance(val, (np.floating, np.float64, np.float32)):
                return float(val)
            return val
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input': dict(request.data),
            'prediction': readable_labels.get(class_label_str, class_label_str),  # Human-readable label
            'predicted_class': class_label_str,  # String class label
            'confidence': round(float(probabilities[to_py(prediction)]) * 100, 1)
        }
        for key, value in record.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                record[key] = None
        try:
            json.dumps(record)
        except TypeError as e:
            print(f"[ERROR] Record not JSON serializable: {record}")
        try:
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                except json.JSONDecodeError:
                    history = []
            else:
                history = []
            history.insert(0, record)
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] Failed to save prediction history: {str(e)}")
        return Response({
            'prediction': readable_labels.get(class_label_str, class_label_str),  # Human-readable label
            'predicted_class': class_label_str,  # String class label
            'probability': float(probabilities[to_py(prediction)]),
            'confidence': round(float(probabilities[to_py(prediction)]) * 100, 1),
            'class_probabilities': {
                str(encoder.inverse_transform([i])[0]): float(prob)
                for i, prob in enumerate(probabilities)
            }
        })
    except Exception as e:
        import traceback
        print("PREDICT ERROR:", e)
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
def train(request):
    """Trigger training pipeline (async)"""
    from .tasks import run_training_pipeline  # Celery task
    
    try:
        # Trigger async training
        task = run_training_pipeline.delay()
        return Response({
            'status': 'Training started',
            'task_id': task.id,
            'monitor_url': f'/tasks/{task.id}/'
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
def get_all_rules(request):
    """Get all association rules (raw, no formatting, for modal speed)"""
    import time
    t0 = time.time()
    rules = load_association_rules()
    # Map plural to singular for frontend compatibility
    result = []
    for _, row in rules.iterrows():
        result.append({
            'antecedent': row.get('antecedents', ''),
            'consequent': row.get('consequents', ''),
            'support': row.get('support', None),
            'confidence': row.get('confidence', None),
            'lift': row.get('lift', None),
            # Include other fields if needed
        })
    t1 = time.time()
    print(f"[get_all_rules] Loaded {len(result)} rules in {t1-t0:.2f}s (singular fields)")
    return Response(result)

@api_view(['GET'])
def get_all_itemsets(request):
    """Get all frequent itemsets"""
    itemsets_path = os.path.join(settings.BASE_DIR, 'data', 'frequent_itemsets.csv')
    itemsets = pd.read_csv(itemsets_path).replace([np.nan, np.inf, -np.inf], None)
    return Response(itemsets.to_dict(orient='records'))

@api_view(['GET'])
def all_students(request):
    """Return all students as JSON for modal view (example: from training data)"""
    import pandas as pd
    import os
    from django.conf import settings
    try:
        data_path = os.path.join(settings.BASE_DIR, 'ModifiedFinalData.xlsx')
        df = pd.read_excel(data_path)
        # Example: return index and a name/ID column if available
        students = []
        for idx, row in df.iterrows():
            students.append({
                'id': idx + 1,
                'name': row.get('Name', f'Student {idx+1}')
            })
        return Response(students)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
def prediction_history(request):
    """Get prediction history from file with error handling"""
    import os
    import json
    import time
    from django.conf import settings
    
    t0 = time.time()
    history_path = os.path.join(settings.BASE_DIR, 'prediction_history.json')
    
    history = []  # Default empty history
    
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                # Attempt to load the JSON file
                history = json.load(f)
                
        except json.JSONDecodeError as e:
            # Handle corrupted JSON file
            print(f"[ERROR] Invalid JSON in history file: {str(e)}")
            
            # Try to recover by reading line-by-line
            history = []
            with open(history_path, 'r') as f:
                for line in f:
                    try:
                        # Attempt to parse each line as JSON
                        record = json.loads(line)
                        history.append(record)
                    except json.JSONDecodeError:
                        # Skip invalid lines
                        continue
            
            # Rewrite the file with valid JSON
            try:
                with open(history_path, 'w') as f:
                    json.dump(history, f, indent=2)
                print(f"[INFO] Recovered {len(history)} valid records from corrupted history file")
            except Exception as e:
                print(f"[ERROR] Failed to rewrite history file: {str(e)}")
                
        except Exception as e:
            print(f"[ERROR] Failed to load history: {str(e)}")
    
    t1 = time.time()
    print(f"[prediction_history] Loaded {len(history)} records in {t1-t0:.4f}s")
    return Response(history)