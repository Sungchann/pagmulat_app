from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import joblib
import pickle 
import json
from .process_response import process_response

@api_view(['POST'])
def assessment_predict(request):

    try:
    
        survey_data = request.data

        processed_data = process_response(survey_data)
        
        user_features = set([key for key, value in processed_data.items() if value == 1])

        # classifier = joblib.load('best_classifier_model.joblib')

        # feature_list = [processed_data[col] for col in sorted(processed_data.keys())]
        # prediction_value = classifier.predict([feature_list])[0]
        # prediction_confidence = max(classifier.predict_proba([feature_list])[0]) * 100

        # with open('association_rules.pkl', 'rb') as f:
        #     association_rules = pickle.load(f)

        # relevant_rules = []
        # for index, row in association_rules.iterrows():
        #     antecedents = set(row['antecedents'])
        #     if antecedents.issubset(user_features):
        #         relevant_rules.append({
        #             'antecedent': ', '.join(list(antecedents)),
        #             'consequent': ', '.join(list(row['consequents'])),
        #             'support': row['support'],
        #             'confidence': row['confidence'],
        #             'lift': row['lift']
        #         })
        
        # response_data = {
        #     'prediction': prediction_value,
        #     'confidence': f"{prediction_confidence:.2f}%",
        #     'predicted_class': 'Some_Predicted_Class',
        #     'arm_rules': relevant_rules
        # }

        return Response(user_features, status=status.HTTP_200_OK)

    except FileNotFoundError as e:
        return Response({'error': f"Model file not found: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)