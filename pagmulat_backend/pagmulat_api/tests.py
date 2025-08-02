from django.test import TestCase
from .models import StudentResponse
from .arm_processing import preprocess_full_dataset, split_train_test

class BehaviorAnalysisTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Create test data
        StudentResponse.objects.create(
            year='3', program='IT', age_group='21-23',
            lms_usage='Often', productivity='Yes', is_real=True
        )
    
    def test_response_creation(self):
        response = StudentResponse.objects.get(id=1)
        self.assertEqual(response.program, "IT")
        self.assertTrue(response.is_real)
    
    def test_arm_processing(self):
        # Test ARM pipeline integration
        df = preprocess_full_dataset("ModifiedFinalData.xlsx")
        self.assertGreater(len(df), 40)
        
        X_train, X_test, y_train, y_test = split_train_test(
            df, target_column="Productive_Yes"
        )
        self.assertGreater(len(X_train), 30)
        self.assertGreater(len(X_test), 8)