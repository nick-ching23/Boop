# Boop: AI-Driven Mental Health Monitoring

Empowering digital well-being with AI-driven insights into mental health risks on social media.

## Overview

Boop is an innovative mental health platform designed to identify and support individuals showing signs of self-harm and suicidal ideation on social media. Utilizing cutting-edge natural language processing (NLP) through a fine-tuned BERT model, Boop analyzes social media content to detect emotional distress with high accuracy. Integrated with Reddit via APIs, Boop offers a proactive approach to mental health monitoring and intervention.

## Features
- **AI-Powered Analysis:** Leverages a finely-tuned BERT model for advanced sentiment analysis, achieving 94% accuracy in detecting potential self-harm or suicidal thoughts.
- **Analysis of Reddit Comments:** Utilizes [reddit-user-to-sqlite](https://github.com/xavdid/reddit-user-to-sqlite/?tab=readme-ov-file) to pull Reddit comment and post data for analysis 
- **User-Friendly Interface:** A Flask-built application paired with a dynamic frontend, ensuring ease of access and interaction for users seeking mental health insights.

## Achievements
- Awarded "Best Project in Mental Hack" for enhancing digital well-being and innovative use of machine learning for mental health support. 
- Recognized for its innovative use of AI and API integrations to facilitate immediate support and intervention for individuals exhibiting signs of mental distress.

## Demo
[Technical Demo](https://youtu.be/Yrkgv3PpxnY)

## Installation
To set up Boop for development or personal use, follow these steps:

**1. clone git repo**
```
git clone https://github.com/nick-ching23/boop.git
cd boop
```

**2. Download the model binary and move it to 'boop' folder**
- [Google Drive Model Link](https://drive.google.com/file/d/1X9agZvBPm0p34b6zIUUjo1Zw_owoar5d/view?usp=sharing)
- Move model_v2.pth to the folder 'boop'

**2. Set Up a Virtual Environment**
```
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
.\venv\Scripts\activate   # On Windows
```
**3. Install Dependencies**
```
pip install -r requirements.txt
```

**4. Run Application**
```
python3 main.py
```
## Usage 

After launching Boop, navigate to http://localhost:5000 in your web browser to access the application. Users can interact with the platform to understand the AI's analysis and receive guidance on potential intervention strategies.

## License
Boop is released under the MIT License. See the LICENSE file for more details.
