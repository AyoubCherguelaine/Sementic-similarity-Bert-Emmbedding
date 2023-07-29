# BERT Movie Search Engine

The BERT Movie Search Engine is a Python-based tool that allows you to search for similar movies based on the provided query using BERT embeddings and cosine similarity. It utilizes pre-trained BERT models for computing embeddings and then compares the query with movie plot summaries to find similar movies.

## Requirements

Before using the BERT Movie Search Engine, make sure you have the following libraries installed:

- NumPy
- pandas
- scikit-learn (for cosine_similarity)
- transformers (for BERT model and tokenizer)
- FastAPI for web server hosting
- Uvicorn to run the API

You can install the required packages using `pip`:

```bash
pip install numpy pandas scikit-learn transformers fastapi uvicorn
```

## Usage

1. **Model Setup**: The movie search engine requires the `Model.py` file containing the functions `get_bert_embeddings`, `tokenizer`, and `model` for generating BERT embeddings. Make sure to import these functions into the main script for the search engine to work correctly.

2. **Prepare the DataFrame**: Ensure that you have a CSV file containing movie data with columns "Release Year," "Title," "Genre," and "Plot." You can read the CSV file using pandas and select the relevant columns to create a DataFrame:

```python
import pandas as pd

df = pd.read_csv("movies.csv")
df = df[['Release Year', 'Title', 'Genre', 'Plot']]
df['Index'] = df.apply(Prepare_Index, axis=1)
```

3. **Initialize the BERTSearchEngine**: Create an instance of the `BERTSearchEngine` class with the prepared DataFrame:

```python
from BERTSearchEngine import BERTSearchEngine

SearchEngineBert = BERTSearchEngine(df)
```

4. **Search for Similar Movies**: Use the `Search` function to find similar movies based on a user-provided query. The function takes the search query, the initialized `BERTSearchEngine` instance, and optional filters for genre and release year:

```python
from BERTSearchEngine import Search

# Perform the search
result = Search(df, SearchEngineBert, genre="Action", year=2021)
print(result)
```

The search engine will return a DataFrame containing the most similar movies based on the query, sorted by similarity score in descending order.

## Functions

### `add_similarity_scores_to_df(df, similarity_dict)`

This function takes a DataFrame (`df`) and a similarity dictionary (`similarity_dict`) as input. It adds a "similarity" column to the DataFrame, containing the similarity scores for each movie in the `similarity_dict`. The DataFrame is then sorted based on similarity scores and returned.

### `BERTSearchEngine`

This class represents the BERT-based search engine. It takes a DataFrame as input during initialization and computes BERT embeddings for the movie plot summaries. The `search` method allows you to perform similarity searches based on a given query and returns a DataFrame with search results.

### `Search(df, SearchEngineBert, genre=None, year=None)`

This function takes the DataFrame (`df`), an instance of the `BERTSearchEngine` (`SearchEngineBert`), and optional filters for genre (`genre`) and release year (`year`). It prompts the user to enter a movie name as the search query, performs the similarity search, and returns a DataFrame with the most similar movies based on the query, optionally filtered by genre and release year.

## Example

```python
from BERTSearchEngine import test

# Execute the test function
results = test()
print(results)
```

The `test` function reads the movie data from a CSV file, initializes the BERT search engine, and performs a search for similar movies based on a user-provided query.

Please make sure to customize the file paths and data according to your specific setup.

Happy movie searching! 🎬

## Note 
this documentation writed by chat GPT