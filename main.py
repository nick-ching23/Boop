import numpy
from flask import Flask, request, render_template, render_template_string,jsonify,abort
import subprocess
import sqlite3
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
import math

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
model.load_state_dict(torch.load('model_v2.pth'))  # Load the fine-tuned weights
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

@app.route('/', methods=['GET'])
def home():
    # Render the home page with the form
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    results = None
    if request.method == 'POST':
        try:
            reddit_id = request.form['reddit_id']
        except KeyError:
        # If 'reddit_id' was not found in the form data, return a 400 Bad Request response
            abort(400, description="Missing reddit_id")
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
            cursor.execute("select text from comments order by timestamp DESC limit 5")
            comments = cursor.fetchall()

            cursor.execute("select text from posts order by timestamp DESC limit 5")
            posts = cursor.fetchall()

            conn.close()

            subprocess.run(["rm","-rf", db_file], check=True) 
            text = ""
            inp_txt = ""

            for c in comments:
                if len(c[0].split(" ")) <= 3:
                    continue
                text += (c[0] + ". ")
                inp_txt += ("-- " + c[0] + ".<br><br>")

            for p in posts:
                if len(p[0].split(" ")) <= 3:
                    continue
                text += (p[0] + ". ")
                inp_txt += ("--" + p[0] + ".<br><br>")

            text = remove_emojis(text)
            text = text.replace('\n','')
            predicted_class, probabilities = predict_single_text(model, tokenizer, text)
            probabilities *= 100
            results = {"latest_post":inp_txt,"probability":probabilities,"left":math.floor((probabilities[0]*12)/100),"right":(math.floor((probabilities[1]*12)/100))}

        except sqlite3.Error as e:
            return jsonify({"error": "Failed to read from the database"}), 500

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
