from fastapi import FastAPI
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

class BERTSearchEngine:
    def __init__(self, df):
        self.documents = df
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.embeddings = self.get_bert_embeddings()

    def get_bert_embeddings(self):
        inputs = self.tokenizer(list(self.documents['Title'] + ": " + self.documents['Plot']), return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.pooler_output.detach().numpy()
        return embeddings

    def search(self, query):
        if isinstance(query, str):
            query = [query]

        query_inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        query_outputs = self.model(**query_inputs)
        query_embedding = query_outputs.pooler_output.detach().numpy()

        similarities = cosine_similarity(query_embedding, self.embeddings)
        similar_indices = np.argsort(similarities[0])[::-1]
        SimRes = [[i, similarities[0][i]] for i in similar_indices]
        return dict(SimRes)

def prepare_index(row):
    return row['Title'] + ":\n" + row['Plot']

def add_similarity_scores_to_df(df, similarity_dict):
    df['similarity'] = df.index.map(lambda x: similarity_dict.get(x, 0.0))
    df_sorted = df.sort_values(by='similarity', ascending=False)
    return df_sorted

df = pd.read_csv("wiki_movie_plots_deduped.csv")
df = df[['Release Year', 'Title', 'Genre', 'Plot']]
df['Index'] = df.apply(prepare_index, axis=1)
search_engine_bert = BERTSearchEngine(df)

@app.get("/search/")
def movie_search(movie_name: str, genre: str = None, year: int = None):
    res = search_engine_bert.search(movie_name)
    sorted_result = add_similarity_scores_to_df(df.copy(), res)

    if genre:
        sorted_result = sorted_result[sorted_result['Genre'] == genre]
    if year:
        sorted_result = sorted_result[sorted_result['Release Year'] == year]

    return sorted_result.to_dict(orient='records')
