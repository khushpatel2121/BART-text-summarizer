from flask import Flask, request, jsonify
from transformers import BartTokenizer, BartForConditionalGeneration
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Load the BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text', '')

    # Tokenize the input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Create response with CORS headers
    response = jsonify({'summary': summary})
    response.headers.add('Access-Control-Allow-Origin', '*')  # Adjust origin as needed

    return response

if __name__ == '__main__':
    app.run(debug=True)
