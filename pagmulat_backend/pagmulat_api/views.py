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
    try:
        rules = pd.read_csv(rules_path)
        # Replace any NaN, inf, or -inf values with None
        rules = rules.replace([np.nan, np.inf, -np.inf], None)
        
        # Ensure we have the required columns
        required_columns = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
        missing_columns = [col for col in required_columns if col not in rules.columns]
        if missing_columns:
            print(f"Warning: Missing columns in association rules: {missing_columns}")
            print(f"Available columns: {list(rules.columns)}")
        
        return rules
    except Exception as e:
        print(f"Error loading association rules: {e}")
        # Return empty DataFrame with expected structure if file can't be loaded
        return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])

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
        if not os.path.exists(metadata_path):
            return Response({
                'error': 'Training metadata not found. Please train the model first.'
            }, status=404)
            
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Load ARM rules
        rules = load_association_rules()
        
        if rules.empty:
            return Response({
                'error': 'No association rules found. Please train the model first.'
            }, status=404)
        
        # Get top 5 rules by confidence (handle missing columns gracefully)
        if 'confidence' in rules.columns and len(rules) > 0:
            top_rules = rules.sort_values('confidence', ascending=False).head(5)
        else:
            top_rules = rules.head(5) if len(rules) > 0 else pd.DataFrame()
        
        # Format rules for frontend
        formatted_rules = []
        for _, row in top_rules.iterrows():
            try:
                formatted_rules.append({
                    'antecedent': format_itemset(row.get('antecedents', '')),
                    'consequent': format_itemset(row.get('consequents', '')),
                    'support': round(float(row.get('support', 0)), 2),
                    'confidence': round(float(row.get('confidence', 0)), 2),
                    'lift': round(float(row.get('lift', 0)), 2)
                })
            except Exception as e:
                print(f"Error formatting rule: {e}")
                continue
        
        # Prepare metrics with safe fallbacks
        active_students = metadata.get('model', {}).get('data_records', 0)
        arm_section = metadata.get('arm', {})
        
        # Build response
        response_data = {
            'metrics': {
                'active_students': active_students,
                'arm_rules': arm_section.get('num_rules', 0),
                'frequent_itemsets': arm_section.get('num_itemsets', 0),
                'model_accuracy': round(metadata.get('avg_confidence', 0.0), 4)
            },
            'thresholds': {
                'support': {
                    'value': arm_section.get('min_support', 0.1),
                    'display': f"≥ {arm_section.get('min_support', 0.1)}"
                },
                'confidence': {
                    'value': arm_section.get('min_confidence', 0.65),
                    'display': f"≥ {arm_section.get('min_confidence', 0.65)}"
                },
                'lift': {
                    'value': arm_section.get('min_lift', 1.2),
                    'display': f"> {arm_section.get('min_lift', 1.2)}"
                }
            },
            'rules_table': formatted_rules
        }
        
        # Cache for 1 hour
        cache.set(cache_key, response_data, timeout=3600)
        return Response(response_data)
        
    except Exception as e:
        import traceback
        print(f"ARM Dashboard Error: {e}")
        traceback.print_exc()
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
    """ARM-based prediction using association rules"""
    try:
        print("PREDICT INPUT:", request.data)
        
        # Load association rules
        rules_path = os.path.join(settings.BASE_DIR, 'data', 'association_rules_filtered.csv')
        if not os.path.exists(rules_path):
            return Response({
                'error': 'Association rules not found. Please train the model first.',
                'suggestion': 'Run training to generate ARM rules.'
            }, status=404)
        
        rules_df = pd.read_csv(rules_path)
        data = request.data
        
        # Build feature set from survey input for rule matching
        active_features = set()
        
        # Map survey data to feature names (same as training)
        if 'year_level' in data:
            active_features.add(f"Year_{data['year_level'][-1]}")
        
        if 'program' in data:
            prog_map = {'BS Computer Science': 'CS', 'BS Information Technology': 'IT', 'BS Entertainment and Multimedia Computing': 'EMC'}
            prog = prog_map.get(data['program'], data['program'])
            active_features.add(f"Program_{prog}")
        
        if 'age' in data:
            age = int(data['age'])
            if age < 18:
                active_features.add('Age_Under18')
            elif 18 <= age <= 20:
                active_features.add('Age_18_20')
            elif 21 <= age <= 23:
                active_features.add('Age_21_23')
            elif 24 <= age <= 26:
                active_features.add('Age_24_26')
            elif 27 <= age <= 29:
                active_features.add('Age_27_29')
            else:
                active_features.add('Age_30_Plus')
        
        # Map other survey fields
        if 'lms_usage' in data:
            active_features.add(f"LMS_Usage_{data['lms_usage']}")
        if 'code_platform_usage' in data:
            active_features.add(f"Code_Compiler_Usage_{data['code_platform_usage']}")
        if 'ai_usage' in data:
            active_features.add(f"ChatGPT_AI_Usage_{data['ai_usage']}")
        if 'social_media_usage' in data:
            active_features.add(f"Social_Media_Daily_Visit_{data['social_media_usage']}")
        if 'fixed_study_schedule' in data:
            active_features.add(f"Fixed_Study_Schedule_{data['fixed_study_schedule']}")
        if 'study_start_time' in data:
            active_features.add(f"Study_Start_Time_{data['study_start_time']}")
        if 'assignment_timeliness' in data:
            active_features.add(f"Submit_Assignments_On_Time_{data['assignment_timeliness']}")
        if 'collab_tools_usage' in data:
            active_features.add(f"Online_Collaboration_Tools_{data['collab_tools_usage']}")
        if 'sleep_time' in data:
            active_features.add(f"Sleep_Time_Weekdays_{data['sleep_time']}")
        if 'burnout' in data:
            active_features.add(f"Burnout_Exhaustion_{data['burnout']}")
        if 'study_breaks' in data:
            active_features.add(f"Regular_Breaks_Studying_{data['study_breaks']}")
        if 'motivation' in data:
            active_features.add(f"Motivation_{data['motivation']}")
        
        print(f"Active features: {active_features}")
        
        # Find matching rules
        matching_rules = []
        for _, rule in rules_df.iterrows():
            # Parse antecedents (features that must be present)
            antecedents = set(rule['antecedents'].split(', ')) if rule['antecedents'] else set()
            
            # Check if all antecedents are satisfied by active features
            if antecedents.issubset(active_features):
                matching_rules.append({
                    'antecedent': rule['antecedents'],
                    'consequent': rule['consequents'],
                    'confidence': float(rule['confidence']),
                    'lift': float(rule['lift']),
                    'support': float(rule['support'])
                })
        
        # Sort by confidence descending
        matching_rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Determine prediction based on rules
        productivity_prediction = "Unknown"
        confidence = 0.0
        reasoning = []
        
        if matching_rules:
            # Look for productivity-related rules
            productivity_rules = [r for r in matching_rules if 'Productive_' in r['consequent']]
            
            if productivity_rules:
                top_rule = productivity_rules[0]
                consequent = top_rule['consequent']
                confidence = top_rule['confidence']
                
                if 'Productive_Yes' in consequent:
                    productivity_prediction = "Highly Productive"
                elif 'Productive_Sometimes' in consequent:
                    productivity_prediction = "Moderate Productivity"
                elif 'Productive_No' in consequent:
                    productivity_prediction = "Low Productivity"
                
                reasoning.append(f"Based on rule: IF [{top_rule['antecedent']}] THEN [{consequent}] (Confidence: {confidence:.1%})")
            else:
                # No direct productivity rules, make inference from other patterns
                top_rules = matching_rules[:3]
                positive_indicators = 0
                for rule in top_rules:
                    if any(positive in rule['consequent'].lower() for positive in ['always', 'high', 'motivated', 'yes']):
                        positive_indicators += 1
                
                if positive_indicators >= 2:
                    productivity_prediction = "Moderate Productivity"
                    confidence = 0.6
                    reasoning.append(f"Inferred from {positive_indicators} positive behavior patterns")
                else:
                    productivity_prediction = "Uncertain"
                    confidence = 0.4
                    reasoning.append("Insufficient patterns for strong prediction")
        else:
            productivity_prediction = "No Matching Patterns"
            confidence = 0.0
            reasoning.append("No association rules match your responses")
        
        # Save prediction history
        from datetime import datetime
        history_path = os.path.join(settings.BASE_DIR, 'prediction_history.json')
        
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input': dict(request.data),
            'prediction': productivity_prediction,
            'confidence': round(confidence * 100, 1),
            'method': 'Association Rule Mining',
            'matching_rules': len(matching_rules),
            'reasoning': reasoning[:2]  # Top 2 reasons
        }
        
        try:
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                except json.JSONDecodeError:
                    history = []
            else:
                history = []
            
            history.insert(0, record)
            # Keep only last 100 predictions
            history = history[:100]
            
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] Failed to save prediction history: {str(e)}")
        
        return Response({
            'prediction': productivity_prediction,
            'confidence': round(confidence * 100, 1),
            'method': 'Association Rule Mining',
            'matching_rules': len(matching_rules),
            'top_rules': matching_rules[:5],  # Return top 5 matching rules
            'reasoning': reasoning,
            'active_features_count': len(active_features)
        })
        
    except Exception as e:
        import traceback
        print("PREDICT ERROR:", e)
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)
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
        
        # ENGINEERED FEATURES - Add composite features for better distinction
        
        # 1. Positive Study Habits Score (0-6)
        positive_study_score = 0
        if data.get('fixed_study_schedule') == 'Yes': positive_study_score += 1
        if data.get('study_hours') in ['>5', 'More than 5 hours']: positive_study_score += 1
        if data.get('study_start_time') in ['Morning', 'Afternoon']: positive_study_score += 1
        if data.get('assignment_timeliness') == 'Always': positive_study_score += 1
        if data.get('study_breaks') == 'Yes': positive_study_score += 1
        if data.get('sleep_time') in ['Before 10 PM', '10 PM - 12AM']: positive_study_score += 1
        features_dict['positive_study_score'] = positive_study_score
        
        # 2. Negative Habits Score (0-5)
        negative_habits_score = 0
        if data.get('social_media_usage') in ['Always', 'Often']: negative_habits_score += 1
        if data.get('burnout') in ['Always', 'Often']: negative_habits_score += 1
        if data.get('social_media_distraction') in ['Always', 'Often']: negative_habits_score += 1
        if data.get('block_distractions') == 'No': negative_habits_score += 1
        if data.get('sleep_time') == 'After 2 AM': negative_habits_score += 1
        features_dict['negative_habits_score'] = negative_habits_score
        
        # 3. Tool Usage Count
        productivity_tool_count = len(data.get('productivity_tools', []))
        academic_tool_count = len(data.get('academic_tools', []))
        motivation_trigger_count = len(data.get('motivation_triggers', []))
        features_dict['productivity_tool_count'] = min(productivity_tool_count, 5)  # Cap at 5
        features_dict['academic_tool_count'] = min(academic_tool_count, 7)  # Cap at 7
        features_dict['motivation_trigger_count'] = min(motivation_trigger_count, 6)  # Cap at 6
        
        # 4. High Performer Indicator (combines multiple positive signals)
        high_performer_score = 0
        if data.get('motivation') in ['Very motivated', 'Motivated']: high_performer_score += 1
        if data.get('ai_usage') in ['Sometimes', 'Often'] and data.get('ai_usage') != 'Always': high_performer_score += 1  # Balanced AI use
        if data.get('lms_usage') in ['Always', 'Often']: high_performer_score += 1
        if data.get('code_platform_usage') in ['Always', 'Often']: high_performer_score += 1
        if productivity_tool_count >= 2: high_performer_score += 1
        features_dict['high_performer_score'] = high_performer_score
        
        # 5. Balanced Digital Habits (not over-reliant, uses tools well)
        balanced_digital = 0
        if data.get('digital_reliance') == 'No': balanced_digital += 1
        if data.get('block_distractions') == 'Yes': balanced_digital += 1
        if data.get('social_media_distraction') in ['Never', 'Rarely']: balanced_digital += 1
        features_dict['balanced_digital_habits'] = balanced_digital
        
        # Digital Habits/Change/Reflection (free text) are ignored for model
        # Build DataFrame for prediction
        X = pd.DataFrame([features_dict[f] for f in feature_names]).T
        X.columns = feature_names
        print("[DEBUG] Input feature vector:")
        print(X.to_dict(orient='records')[0])
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        print("[DEBUG] Class probabilities:")
        for idx, prob in enumerate(probabilities):
            if 'class_map' in metadata['model']:
                class_map = metadata['model']['class_map']
                class_label = class_map.get(str(idx), class_map.get(int(idx), idx))
            else:
                class_label = encoder.classes_[idx]
            print(f"  {class_label}: {prob:.4f}")
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
                    with open(history_path, 'r', encoding='utf-8') as f:
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
            with open(history_path, 'r', encoding='utf-8') as f:
                # Attempt to load the JSON file
                history = json.load(f)
                
        except json.JSONDecodeError as e:
            # Handle corrupted JSON file
            print(f"[ERROR] Invalid JSON in history file: {str(e)}")
            
            # Try to recover by reading line-by-line
            history = []
            with open(history_path, 'r', encoding='utf-8') as f:
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
                with open(history_path, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2)
                print(f"[INFO] Recovered {len(history)} valid records from corrupted history file")
            except Exception as e:
                print(f"[ERROR] Failed to rewrite history file: {str(e)}")
                
        except Exception as e:
            print(f"[ERROR] Failed to load history: {str(e)}")
    
    t1 = time.time()
    print(f"[prediction_history] Loaded {len(history)} records in {t1-t0:.4f}s")
    return Response(history)

@api_view(['GET'])
def arm_metadata(request):
    """ARM metadata endpoint for Angular frontend"""
    try:
        # Load metadata
        metadata_path = os.path.join(settings.BASE_DIR, 'training_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return Response(metadata)
        
    except FileNotFoundError:
        return Response({
            'error': 'Training metadata not found. Please train the model first.',
            'model': {
                'data_records': 0,
                'total_rules': 0,
                'frequent_itemsets': 0
            },
            'arm': {
                'top_rules': []
            }
        }, status=404)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
def arm_rules(request):
    """ARM rules endpoint for Angular frontend"""
    try:
        # Load association rules
        rules_path = os.path.join(settings.BASE_DIR, 'data', 'association_rules_filtered.csv')
        rules_df = pd.read_csv(rules_path)
        
        # Convert to JSON format for frontend
        rules_list = []
        for _, row in rules_df.iterrows():
            rules_list.append({
                'antecedent': row['antecedents'],
                'consequent': row['consequents'],
                'support': round(float(row['support']), 3),
                'confidence': round(float(row['confidence']), 3),
                'lift': round(float(row['lift']), 2)
            })
        
        return Response(rules_list)
        
    except FileNotFoundError:
        return Response({
            'error': 'Association rules not found. Please train the model first.'
        }, status=404)
    except Exception as e:
        return Response({'error': str(e)}, status=500)