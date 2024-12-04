from django.http import JsonResponse
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('knn_model.pkl')

def classify_lab_event(request):
    try:
        # Extract input parameters from the URL
        value = float(request.GET.get('value', 0))
        valuenum = float(request.GET.get('valuenum', 0))

        # Prepare input data for prediction
        input_data = pd.DataFrame([{
            'value': value,
            'valuenum': valuenum,
        }])

        # Make the prediction
        prediction = model.predict(input_data)[0]

        # Return the classification result (1 = abnormal, 0 = normal)
        return JsonResponse({'knn_model': int(prediction)})

    except Exception as e:
        # Handle errors gracefully
        return JsonResponse({'error': str(e)}, status=400)