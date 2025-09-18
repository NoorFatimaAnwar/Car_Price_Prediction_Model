import numpy as np
import pandas as pd

def predict_car_price(model, scaler, input_data, feature_names):
    
    # Convert dict → DataFrame
    input_df = pd.DataFrame([[
        input_data.get(f, 0) for f in feature_names
    ]], columns=feature_names)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    predicted_price = model.predict(input_scaled)

    return predicted_price[0]
