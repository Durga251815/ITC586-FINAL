# Import necessary libraries
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

# --- Configuration ---
# Assumes the script is run from project/code/
DATA_DIR = os.path.join('data', 'aclImdb')
RESULTS_DIR = os.path.join('results')
DATASET_INFO_DIR = os.path.join(RESULTS_DIR, 'dataset_info') # Specific subdir for these results
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# --- Ensure results directory and dataset_info subdirectory exist ---
os.makedirs(DATASET_INFO_DIR, exist_ok=True)
print(f"Results will be saved in: {DATASET_INFO_DIR}")

# --- Download NLTK data if necessary ---
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

# --- Function to load data from folders ---
def load_imdb_data(data_dir, subset_name):
    """Loads IMDB data from specified train/test directory."""
    texts = []
    labels = []
    sentiments = [] # Store 'positive'/'negative' strings
    # Load positive reviews
    pos_dir = os.path.join(data_dir, 'pos')
    for filename in os.listdir(pos_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
            labels.append(1) # 1 for positive
            sentiments.append('positive')

    # Load negative reviews
    neg_dir = os.path.join(data_dir, 'neg')
    for filename in os.listdir(neg_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                texts.append(f.read())
            labels.append(0) # 0 for negative
            sentiments.append('negative')

    # Create DataFrame
    df = pd.DataFrame({
        'review': texts,
        'sentiment_label': labels,
        'sentiment_text': sentiments,
        'subset': subset_name
    })
    return df

# --- Load Data ---
print("Loading training data...")
df_train = load_imdb_data(TRAIN_DIR, 'train')
print(f"Loaded {len(df_train)} training reviews.")

print("Loading test data...")
df_test = load_imdb_data(TEST_DIR, 'test')
print(f"Loaded {len(df_test)} test reviews.")

# Combine into a single DataFrame for overall analysis
df_all = pd.concat([df_train, df_test], ignore_index=True)
print(f"Total reviews loaded: {len(df_all)}")

# --- Basic Cleaning ---
print("Cleaning text (removing HTML tags, converting to lowercase)...")
def clean_text(text):
    # Remove HTML tags
    text = re.sub(re.compile('<.*?>'), ' ', text)
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

df_all['review_cleaned'] = df_all['review'].apply(clean_text)

# --- Calculate Statistics ---
print("Calculating statistics...")
# Overall counts
total_reviews = len(df_all)
train_reviews = len(df_train)
test_reviews = len(df_test)

# Sentiment distribution
sentiment_counts = df_all['sentiment_text'].value_counts()
train_sentiment_counts = df_train['sentiment_text'].value_counts()
test_sentiment_counts = df_test['sentiment_text'].value_counts()

# Review length analysis (word count)
df_all['word_count'] = df_all['review_cleaned'].apply(lambda x: len(word_tokenize(x)))
word_count_stats = df_all['word_count'].describe()

# --- Analyze Word Frequency ---
print("Analyzing word frequency...")
# Combine all cleaned reviews into one large text block
all_text = ' '.join(df_all['review_cleaned'])

# Tokenize
tokens = word_tokenize(all_text)

# Remove stop words and short tokens (e.g., length < 3)
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]

# Count word frequencies
word_counts = Counter(filtered_tokens)
most_common_words = word_counts.most_common(30) # Get top 30

# --- Save Summary Statistics ---
summary_file = os.path.join(DATASET_INFO_DIR, 'dataset_summary.txt')
print(f"Saving summary statistics to {summary_file}...")
with open(summary_file, 'w') as f:
    f.write("--- IMDB Dataset Summary ---\n\n")
    f.write(f"Total Reviews: {total_reviews}\n")
    f.write(f"Training Reviews: {train_reviews}\n")
    f.write(f"Test Reviews: {test_reviews}\n\n")

    f.write("Overall Sentiment Distribution:\n")
    f.write(f"  Positive: {sentiment_counts.get('positive', 0)}\n")
    f.write(f"  Negative: {sentiment_counts.get('negative', 0)}\n\n")

    f.write("Training Set Sentiment Distribution:\n")
    f.write(f"  Positive: {train_sentiment_counts.get('positive', 0)}\n")
    f.write(f"  Negative: {train_sentiment_counts.get('negative', 0)}\n\n")

    f.write("Test Set Sentiment Distribution:\n")
    f.write(f"  Positive: {test_sentiment_counts.get('positive', 0)}\n")
    f.write(f"  Negative: {test_sentiment_counts.get('negative', 0)}\n\n")

    f.write("Review Length (Word Count) Statistics:\n")
    f.write(f"  Mean: {word_count_stats['mean']:.2f}\n")
    f.write(f"  Median: {df_all['word_count'].median():.0f}\n") # describe() doesn't always show median
    f.write(f"  Std Dev: {word_count_stats['std']:.2f}\n")
    f.write(f"  Min: {word_count_stats['min']:.0f}\n")
    f.write(f"  Max: {word_count_stats['max']:.0f}\n")
    f.write(f"  25th Percentile: {word_count_stats['25%']:.0f}\n")
    f.write(f"  75th Percentile: {word_count_stats['75%']:.0f}\n\n")

    f.write(f"Top {len(most_common_words)} Most Common Words (after cleaning and stopword removal):\n")
    for word, count in most_common_words:
        f.write(f"  {word}: {count}\n")

# --- Generate and Save Visualizations ---
print("Generating and saving plots...")

# 1. Review Length Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_all['word_count'], bins=50, kde=True)
plt.title('Distribution of Review Lengths (Word Count)')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.xlim(0, 1500) # Limit x-axis for better visibility, adjust if needed
plt.tight_layout()
plot_path = os.path.join(DATASET_INFO_DIR, 'review_length_distribution.png')
plt.savefig(plot_path)
print(f"Saved review length plot to {plot_path}")
plt.close() # Close the plot to free memory

# 2. Most Common Words Bar Chart
common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
plt.figure(figsize=(12, 8))
sns.barplot(x='Frequency', y='Word', data=common_words_df, palette='viridis')
plt.title(f'Top {len(most_common_words)} Most Common Words')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.tight_layout()
plot_path = os.path.join(DATASET_INFO_DIR, 'most_common_words.png')
plt.savefig(plot_path)
print(f"Saved common words plot to {plot_path}")
plt.close()

# 3. Sentiment Distribution Pie Chart
plt.figure(figsize=(8, 8))
df_all['sentiment_text'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
plt.title('Overall Sentiment Distribution')
plt.ylabel('') # Hide the y-label
plt.tight_layout()
plot_path = os.path.join(DATASET_INFO_DIR, 'sentiment_distribution_pie.png')
plt.savefig(plot_path)
print(f"Saved sentiment distribution plot to {plot_path}")
plt.close()


print("\nDataset analysis script finished.")