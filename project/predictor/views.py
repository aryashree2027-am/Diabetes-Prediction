from django.shortcuts import render
from .model import train_model

model = train_model()

def home(request):
    if request.method == "POST":
        data = [
            float(request.POST['Glucose']),
            float(request.POST['BloodPressure']),
            float(request.POST['SkinThickness']),
            float(request.POST['Insulin']),
            float(request.POST['BMI']),
            float(request.POST['Age'])
        ]

        prediction = model.predict([data])[0]

        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return render(request, "home.html", {"result": result})

    return render(request, "home.html")
