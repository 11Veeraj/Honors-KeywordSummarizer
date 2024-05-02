# -*- coding: utf-8 -*-
"""

"""

from flask import Flask, render_template, request, session, redirect, url_for


app = Flask(__name__)
app.secret_key = 'your_secret_key'


import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelWithLMHead




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    detected_classes = request.form.getlist('detected_classes')
    if not detected_classes:
        return "No detected classes provided!"
    detected_classes = set(detected_classes)
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import quote

    def search_wikipedia(keywords):
        search_query = ' '.join(keywords)
        encoded_query = quote(search_query)
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={encoded_query}&srprop=snippet&utf8="
        r = requests.get(search_url)
        data = r.json()

        # Extract the URLs of the top three search results
        result_urls = []
        search_results = data['query']['search'][:10]
        for result in search_results:
            page_id = result['pageid']
            result_url = f"https://en.wikipedia.org/?curid={page_id}"
            result_urls.append(result_url)

        return result_urls

    def extract_paragraph_with_keywords(url, keywords):
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')
        paragraphs = soup.find_all('p')
        
        for paragraph in paragraphs:
            paragraph_text = paragraph.get_text(strip=True)
            if all(keyword.lower() in paragraph_text.lower() for keyword in keywords):
                return paragraph_text

        return None

    # Example usage:
    #search_keywords = ["Krishna","Vasudeva","Birth"]
    search_keywords = detected_classes
    # Combine the keywords into a single search query
    search_query = ' '.join(search_keywords)

    # Perform the search and get the top three result URLs
    result_urls = search_wikipedia(search_keywords)
    
    # Extract and store the paragraph containing all the keywords for each result URL
    paragraphs = []
    for url in result_urls:
       paragraph_text = extract_paragraph_with_keywords(url, search_keywords)
       if paragraph_text:
           paragraphs.append(paragraph_text)
           
    # Store the paragraphs in a session variable
    session['paragraphs'] = paragraphs
           
    # Return the detection result and detected classes to the result.html template
    return render_template('result.html', detected_classes=detected_classes, paragraphs=paragraphs)


@app.route('/summarise', methods=['POST'])
def summarise():
    paragraphs = session.get('paragraphs', [])
    session.pop('paragraphs', None)
    
    selected_tokenizer = request.form['selected_tokenizer']
    selected_model = request.form['selected_model']
    
    model_map = {
        't5-base': ('T5-base', 'T5-base'),
        'bert-large-uncased': ('bert-large-uncased', 'bert-large-uncased'),
        'gpt2': ('gpt2', 'gpt2'),
    }
    
    tokenizer_name, model_name = model_map.get(selected_tokenizer, ('T5-base', 'T5-base'))
    
    # Load the selected tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelWithLMHead.from_pretrained(model_name, return_dict=True)
    
    summaries = []
    for i in paragraphs:
        sequence = i
        inputs = tokenizer.encode("summarize: " + sequence, return_tensors='pt', max_length=512, truncation=True)
        output = model.generate(inputs, min_length=30, max_length=100)
        summary = tokenizer.decode(output[0])
        summary = summary[6:-4]
        summaries.append(summary)
    
    return render_template('summarise.html', summaries=summaries)




    
if __name__ == '__main__':
    app.run()
