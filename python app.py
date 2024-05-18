from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the pickled chatbot
with open('diabetes_chatbot.pkl', 'rb') as f:
    chatbot = pickle.load(f)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = chatbot.respond(user_input)  # Assuming chatbot has a respond method
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)


