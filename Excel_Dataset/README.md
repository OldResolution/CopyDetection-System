# 📊 Processed Books Dataset

Welcome to the **Excel Dataset** directory. This directory contains the processed and extracted text data derived from the raw books corpus, which is used for analysis and training in the CopyDetection-System.

## 📁 Files Included

- `processed_books_dataset-1.xlsx`: The primary dataset containing text segments and pre-calculated linguistic metrics.

## 📝 Dataset Structure

The dataset contains the text content of the books broken down by paragraph, along with associated metadata and lexical features. 

The columns in this dataset are:

| Column | Description |
| :--- | :--- |
| `book_id` | A unique identifier for the specific book segment (e.g., `Science_Fiction_JRR_Tolkien_The_Lord_of_the_Rings_p1`). |
| `genre` | The genre of the book (e.g., `Science Fiction`). |
| `author` | The author of the book (e.g., `JRR_Tolkien`). |
| `book_title` | The title of the book. |
| `paragraph_num` | The sequential index of the paragraph/segment within the text. |
| `text_content` | The actual raw text content of the paragraph. |
| `sentence_count` | The total number of sentences in this paragraph. |
| `word_count` | The total number of words in this paragraph. |
| `avg_sentence_length` | The average number of words per sentence. |
| `avg_word_length` | The average character length of the words used. |
| `type_token_ratio` | The ratio of unique words (types) to total words (tokens), representing lexical richness/diversity. |

## 🛠️ Usage

This dataset provides the foundational data for NLP feature extraction. The pre-calculated metrics (`avg_sentence_length`, `type_token_ratio`, etc.) are particularly useful for stylometric analysis to identify authorship or detect if a text has been copied or paraphrased from these sources.
