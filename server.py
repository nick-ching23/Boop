import numpy
from flask import Flask, request, render_template_string,jsonify
import subprocess
import sqlite3
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re

def remove_emojis(text):
    cleaned_text = ""
    emoji = False
    for s in text:
        if not s.isascii():
            emoji = True
            continue
        if s == " ":
            emoji = False
        if emoji:
            continue
        cleaned_text += s
    return cleaned_text

app = Flask(__name__)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Instantiate the model architecture
model.load_state_dict(torch.load('model.pth'))  # Load the fine-tuned weights
model.eval()  # Set the model to evaluation mode

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_single_text(model, tokenizer, text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Move tensors to the same device as model
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Apply softmax to logits to get probabilities
    probabilities = torch.softmax(logits, dim=-1)
    
    # Get the predicted class (0 or 1) based on the highest probability
    predicted_class = torch.argmax(probabilities, dim=-1).cpu().numpy()[0]  # Extract the predicted class
    
    # Optionally, convert probabilities to numpy for easier interpretation
    probabilities = probabilities.cpu().numpy()[0]
    
    return predicted_class, probabilities

# Example usage

# HTML template for the web page
HTML_TEMPLATE = """
<!doctype html>
<html>
<head><title>Reddit ID Processor</title></head>
<body>
  <h2>Enter Reddit ID</h2>
  <form method="post">
    <input type="text" name="reddit_id" />
    <input type="submit" value="Submit" />
  </form>
  {% if model_output %}
    <h3>Model Output:</h3>
    <p>{{ model_output }}</p>
  {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    model_output = None
    if request.method == 'POST':
        reddit_id = request.form['reddit_id']
        # Here you would process the Reddit ID with your model
        # For demonstration, we'll just echo the Reddit ID as the model's output
        db_file = 'my-reddit-data.db'

        # Run the external command
        try:
            subprocess.run(["reddit-user-to-sqlite", "user", reddit_id, "--db", db_file], check=True)
        except subprocess.CalledProcessError as e:
            return jsonify({"error": "Failed to fetch data from Reddit"}), 500

        # Connect to the SQLite database and query for comments and posts
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            # Assuming the tables are named 'comments' and 'posts'
            # Adjust SQL queries based on actual table schema
            cursor.execute("select text from comments order by timestamp DESC limit 101")
            comments = cursor.fetchall()

            cursor.execute("select text from posts order by timestamp DESC limit 101")
            posts = cursor.fetchall()

            conn.close()

            subprocess.run(["rm","-rf", db_file], check=True) 
            text = ""
            comment_results = []
            posts_results = []
            for c in comments:
                if len(c[0].split(" ")) <= 3:
                    continue
                text += (c[0] + ". ")
                c_predicted_class, c_probabilities = predict_single_text(model, tokenizer, c[0])
                comment_results.append(c[0]+" "+str(c_predicted_class)+ str(c_probabilities))
            for p in posts:
                if len(p[0].split(" ")) <= 3:
                    continue
                text += (p[0] + ". ")
                p_predicted_class, p_probabilities = predict_single_text(model, tokenizer, p[0])
                posts_results.append(p[0]+" "+str(p_predicted_class) + str(p_probabilities))

            text = remove_emojis(text)
            text = text.replace('\n','')
            predicted_class, probabilities = predict_single_text(model, tokenizer, text)

            return jsonify({"Input":text,"Predicted class": str(predicted_class), "Probability of suicide":str(probabilities),"Probability of no-suicide":str(probabilities[1]),"CR":comment_results,"PR":posts_results})

        except sqlite3.Error as e:
            return jsonify({"error": "Failed to read from the database"}), 500
        model_output = f"Processed output for Reddit ID: {reddit_id}"
    return render_template_string(HTML_TEMPLATE, model_output=model_output)

if __name__ == "__main__":
    app.run(debug=True)
