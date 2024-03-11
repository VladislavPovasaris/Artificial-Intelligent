import tkinter as tk
from tkinter import messagebox, scrolledtext
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import numpy as np

nltk.download('wordnet')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('arxiv_paper1s.csv')

# Preprocess the text data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    text = re.sub(r'^[a-z]\s+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['processed_text'] = df['title'] + ' ' + df['abstract']
df['processed_text'] = df['processed_text'].apply(preprocess_text)

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['processed_text'])

# Perform topic modeling using LDA
lda = LatentDirichletAllocation(n_components=5, random_state=0, n_jobs=-1) # Adjust parameters as needed
doc_topic_distribution = lda.fit_transform(X)

# Function to display categories with top keywords
def display_categories():
    # Create the categories window
    categories_window = tk.Toplevel()
    categories_window.title("Categories and Keywords")

    # Get top keywords for each topic
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords_idx = topic.argsort()[:-11:-1]
        top_keywords = [feature_names[i] for i in top_keywords_idx]

        # Create a frame for each topic
        topic_frame = tk.Frame(categories_window)
        topic_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a label for the topic with top keywords as the text
        topic_label = tk.Label(topic_frame, text=f"Topic {topic_idx + 1}: {', '.join(top_keywords)}")
        topic_label.pack(side=tk.TOP, fill=tk.X)

        # Create a ScrolledText widget to display the articles for this topic
        articles_text = scrolledtext.ScrolledText(topic_frame, height=10, width=80, wrap=tk.WORD)
        articles_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Find articles most associated with this topic
        top_articles_indices = np.argsort(doc_topic_distribution[:, topic_idx])[::-1][:10]
        for idx in top_articles_indices:
            articles_text.insert(tk.END, f"Title: {df.iloc[idx]['title']}\n")
            articles_text.insert(tk.END, f"Abstract: {df.iloc[idx]['abstract']}\n")
            articles_text.insert(tk.END, f"Published: {df.iloc[idx]['published']}\n")
            articles_text.insert(tk.END, f"Authors: {df.iloc[idx]['authors']}\n")
            articles_text.insert(tk.END, f"URL: {df.iloc[idx]['url']}\n")
            articles_text.insert(tk.END, "---\n")

# GUI-based search
def perform_search():
    search_phrase = entry.get()
    if not search_phrase.strip():
        messagebox.showerror("Error", "Please enter a search phrase.")
        return

    # Perform a dummy search for demonstration purposes
    messagebox.showinfo("Search", "This is a dummy search. No actual search functionality implemented.")

# Create the main window
root = tk.Tk()
root.title("Article Clustering and Search")

# Create a search entry field
entry = tk.Entry(root, width=50)
entry.pack(side=tk.TOP, padx=10, pady=10)

# Create a search button
search_button = tk.Button(root, text="Search", command=perform_search)
search_button.pack(side=tk.TOP, padx=10, pady=10)

# Create a button to display categories
categories_button = tk.Button(root, text="Display Categories", command=display_categories)
categories_button.pack(side=tk.TOP, padx=10, pady=10)

# Start the GUI loop
root.mainloop()
