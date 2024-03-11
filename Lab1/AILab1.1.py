import tkinter as tk
from tkinter import messagebox, scrolledtext
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
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

# Perform clustering to categorize into five groups
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
df['group'] = kmeans.labels_

# Function to perform search
def search(phrase, n_results=30):
    # Convert the input phrase to a vector
    phrase_vector = vectorizer.transform([preprocess_text(phrase)])
    
    # Calculate cosine similarity between the phrase vector and all document vectors
    similarities = cosine_similarity(phrase_vector, X).flatten()
    
    # Get the indices of the most similar documents
    related_docs_indices = similarities.argsort()[::-1][:n_results]  # Sort in descending order and limit to n_results
    
    # Return the most similar articles and their similarities
    return df.iloc[related_docs_indices], similarities[related_docs_indices]

# Function to display categories with top keywords
def display_categories():
    # Hide the main window
    root.withdraw()
    
    # Create the categories window
    categories_window = tk.Toplevel()
    categories_window.title("Categories and Keywords")
    
    # Keep track of keywords that have already been used in other categories
    used_keywords = set()
    
    # Sort the groups in ascending order
    for group_number in sorted(df['group'].unique()):
        group_frame = tk.Frame(categories_window)
        group_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        group_articles = df[df['group'] == group_number]
        
        # Extract top keywords for the category
        group_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        group_X = group_vectorizer.fit_transform(group_articles['processed_text'])
        feature_array = np.array(group_vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(group_X.toarray()).flatten()[::-1]
        
        # Filter out keywords that have already been used in other categories
        top_keywords = [word for word in feature_array[tfidf_sorting] if word not in used_keywords][:10]
        
        # Update the used_keywords set with the keywords used in this category
        used_keywords.update(top_keywords)
        
        # Create a label for the group with the top keywords as the text
        group_label = tk.Label(group_frame, text=f"Group {group_number +1}: {', '.join(top_keywords)}")
        group_label.pack(side=tk.TOP, fill=tk.X)
        
        # Create a ScrolledText widget to display the articles
        articles_text = scrolledtext.ScrolledText(group_frame, height=10, width=80, wrap=tk.WORD)
        articles_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        for index, row in group_articles.iterrows():
            articles_text.insert(tk.END, f"Title: {row['title']}\n")
            articles_text.insert(tk.END, f"Abstract: {row['abstract']}\n")
            articles_text.insert(tk.END, f"Published: {row['published']}\n")
            articles_text.insert(tk.END, f"Authors: {row['authors']}\n")
            articles_text.insert(tk.END, f"URL: {row['url']}\n")
            articles_text.insert(tk.END, "---\n")
    
    # Create a button to go back to the main window
    back_button = tk.Button(categories_window, text="Back to Search", command=lambda: [categories_window.destroy(), root.deiconify()])
    back_button.pack(side=tk.BOTTOM, padx=10, pady=10)
    
    # Start the categories window's main loop
    categories_window.mainloop()

# GUI-based search
def perform_search():
    search_phrase = entry.get()
    if not search_phrase.strip():
        messagebox.showerror("Error", "Please enter a search phrase.")
        return
    
    results, similarities = search(search_phrase)
    result_text.delete(1.0, tk.END)
    for i, (index, row) in enumerate(results.iterrows()):
        result_text.insert(tk.END, f"Title: {row['title']}\n")
        result_text.insert(tk.END, f"Abstract: {row['abstract']}\n")
        result_text.insert(tk.END, f"Published: {row['published']}\n")
        result_text.insert(tk.END, f"Authors: {row['authors']}\n")
        result_text.insert(tk.END, f"URL: {row['url']}\n")
        result_text.insert(tk.END, f"Group: {row['group']}\n")
        result_text.insert(tk.END, f"Similarity: {similarities[i]}\n")
        result_text.insert(tk.END, "---\n")

# Create the main window
root = tk.Tk()
root.title("Article Clustering and Search")

# Create a search entry field
entry = tk.Entry(root, width=50)
entry.pack(side=tk.TOP, padx=10, pady=10)

# Create a search button
search_button = tk.Button(root, text="Search", command=perform_search)
search_button.pack(side=tk.TOP, padx=10, pady=10)

# Create a text widget to display search results
result_text = scrolledtext.ScrolledText(root, height=20, width=80, wrap=tk.WORD)
result_text.pack(side=tk.TOP, padx=10, pady=10)

# Create a button to display categories
categories_button = tk.Button(root, text="Display Categories", command=display_categories)
categories_button.pack(side=tk.TOP, padx=10, pady=10)

# Start the GUI loop
root.mainloop()
