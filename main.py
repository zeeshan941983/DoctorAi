from flask import Flask, request, jsonify
import cv2
import pytesseract
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model for prediction
model_path = "random_forest_model.pkl"  # Replace with the path to your trained model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the pickled chatbot
with open('diabetes_chatbot.pkl', 'rb') as f:
    chatbot = pickle.load(f)

def extract_data_from_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform OCR to extract text
    extracted_text = pytesseract.image_to_string(gray_image)
    
    # Initialize variables
    inside_table = False
    data_rows = []
    
    # Loop through each line of the extracted text
    for line in extracted_text.split('\n'):
        # Check if the line contains any alphanumeric characters
        if any(c.isalnum() for c in line):
            # If the line contains keywords indicating the start of the table
            if "test" in line.lower() and "results" in line.lower():
                inside_table = True
                continue
            # If the line contains keywords indicating the end of the table
            elif "end" in line.lower() and "table" in line.lower():
                inside_table = False
                break
            # If we are inside the table and the line contains data
            elif inside_table:
                columns = line.split()  # Split the line by whitespace
                if len(columns) >= 2:   # Assuming there are at least two columns (test and result)
                    data_rows.append((columns[0], columns[1]))
                    
    # Create a DataFrame from the data rows
    df = pd.DataFrame(data_rows)
    # Transpose the DataFrame to swap rows and columns
    df = df.T
    
    # Convert DataFrame to CSV string
    csv_data = df.to_csv(header=False, index=False)
    
    return csv_data

@app.route('/extract', methods=['POST'])
def extract_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        file_path = 'uploaded_image.jpg'
        file.save(file_path)
        csv_data = extract_data_from_image(file_path)
        return csv_data, 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.json
        input_data = data['input']
        
        # Convert the input data to a numpy array
        input_data_as_numpy_array = np.asarray(input_data)
        
        # Reshape the numpy array as we are predicting for only one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data_reshaped)
        
        # Interpret the prediction
        if prediction[0] == 0:
            prediction_text = "Predict-Diabetic"
        elif prediction[0] == -1:
            prediction_text = "Non-Diabetic"
        elif prediction[0] == 1:
            prediction_text = "Diabetic"
        
        # Return the prediction result
        return jsonify({'prediction': prediction_text})
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = chatbot.respond(user_input)  # Assuming chatbot has a respond method
    return jsonify({'response': response})
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001)

