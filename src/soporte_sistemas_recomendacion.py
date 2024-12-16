# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd

# Para visualización de datos
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt

# Para modelos NLP
# -----------------------------------------------------------------------
import spacy
from nltk.corpus import stopwords
import nltk
import contractions
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity #  Cosine Similarity post Vectorizacion

# nlp=spacy.load("en_core_web_lg")

nltk.download("stopwords")

import re

# Gestionar emojis
# -----------------------------------------------------------------------
from emoji_extractor.extract import Extractor
import emoji

extractor = Extractor()

def limpiar_columna(text):
    stop_words = set(stopwords.words("english"))
    text = str(text)
    text = contractions.fix(text)  # Convierte "don't" -> "do not"

    emojis = extractor.count_emoji(text, check_first=False) # extrae emojis
    emojis = emoji.demojize(" ".join(emojis)) # convierte emojis a texto
    emojis = re.sub(r'[^\w\s]', ' ', emojis) # reemplaza caracteres por espacios
    text = text + " " + emojis # une los emojis transformados a texto con el texto

    # Limpieza de texto
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'[^\w\s]', ' ', text)  # Eliminar puntuación
    text = re.sub(r'\d+', ' ', text)  # Eliminar números
    text = re.sub(r'\s+', ' ', text)  # Reemplazar múltiples espacios o saltos de línea por un espacio
    text = text.strip()  # Quitar espacios en blanco al inicio y al final
    # doc = nlp(text)  # Tokenizar con spaCy
    text = [word for word in text.split()]


    # eliminamos las stop words
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.text.isalpha()]
    return tokens

def generar_bow(df, cleaned_text_column):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df[cleaned_text_column])
    similarity = cosine_similarity(X)
    similarity_matrix = pd.DataFrame(similarity, index=df[cleaned_text_column], columns=df[cleaned_text_column])
    return similarity_matrix

def generar_tfidf(df, cleaned_text_column):
    vectorizer_tfidf = TfidfVectorizer(max_features=10000)
    X_tfidf = vectorizer_tfidf.fit_transform(df[cleaned_text_column])
    similarity = cosine_similarity()
    similarity_matrix = pd.DataFrame(similarity, index=df[cleaned_text_column], columns=df[cleaned_text_column])
    return similarity_matrix

def generar_recomendaciones(top_df_cliente, productos_df, score_column, similarity_matrix, w_score=1, w_similarity=4, n_recomendaciones = 5):
    w_score_ = w_score/(w_score + w_similarity)
    w_similarity_ = w_similarity/(w_score + w_similarity)
    total_similar_products = pd.DataFrame()
    for col in top_df_cliente["tags"].to_list():

        similarity_products = pd.DataFrame(similarity_matrix[col]).reset_index()
        similarity_products.columns = ["tags","similarity"]
        similarity_products["review_score"] = productos_df.loc[productos_df["tags"] == similarity_products["tags"],score_column]

        similarity_products["total_score"] = (similarity_products["review_score"] * w_score_ + similarity_products["similarity"] * w_similarity_)

        total_similar_products = pd.concat([total_similar_products,similarity_products],axis=0)

    total_similar_products = total_similar_products.sort_values(by="total_score", ascending=False).drop_duplicates(subset="tags")
    total_similar_products = total_similar_products[total_similar_products["similarity"]<0.9999999]
    
    return total_similar_products.nlargest(n_recomendaciones, columns=["total_score"])

def get_index_from_title(title, dataframe):
    """
    Obtiene el índice de un dataframe basado en el título de una película.

    Parameters:
    ----------
    title : str
        El título de la película a buscar.
    dataframe : pd.DataFrame
        El dataframe que contiene la información, con una columna 'title'.

    Returns:
    -------
    int
        El índice correspondiente al título de la película en el dataframe.
    """
    return dataframe[dataframe.title == title].index[0]


def get_title_from_index(index, dataframe):
    """
    Obtiene el título de una película basado en su índice en un dataframe.

    Parameters:
    ----------
    index : int
        El índice de la película a buscar.
    dataframe : pd.DataFrame
        El dataframe que contiene la información, con una columna 'title'.

    Returns:
    -------
    str
        El título de la película correspondiente al índice proporcionado.
    """
    return dataframe[dataframe.index == index]["title"].values[0]


def plot(peli1, peli2, dataframe):
    """
    Genera un gráfico de dispersión que compara dos películas en un espacio de características.

    Parameters:
    ----------
    peli1 : str
        Nombre de la primera película a comparar.
    peli2 : str
        Nombre de la segunda película a comparar.
    dataframe : pd.DataFrame
        Un dataframe transpuesto donde las columnas representan películas y las filas características.

    Returns:
    -------
    None
        Muestra un gráfico de dispersión con anotaciones para cada película.
    """
    x = dataframe.T[peli1]     
    y = dataframe.T[peli2]

    n = list(dataframe.columns)    

    plt.figure(figsize=(10, 5))

    plt.scatter(x, y, s=0)      

    plt.title('Espacio para {} VS. {}'.format(peli1, peli2), fontsize=14)
    plt.xlabel(peli1, fontsize=14)
    plt.ylabel(peli2, fontsize=14)

    for i, e in enumerate(n):
        plt.annotate(e, (x[i], y[i]), fontsize=12)  

    plt.show();


def filter_data(df):
    """
    Filtra un dataframe de ratings basado en la frecuencia mínima de valoraciones por película y por usuario.

    Parameters:
    ----------
    df : pd.DataFrame
        Un dataframe con columnas 'movieId', 'userId' y 'rating'.

    Returns:
    -------
    pd.DataFrame
        Un dataframe filtrado que contiene solo las películas con al menos 300 valoraciones 
        y los usuarios con al menos 1500 valoraciones.
    """
    ## Ratings Per Movie
    ratings_per_movie = df.groupby('movieId')['rating'].count()
    ## Ratings By Each User
    ratings_per_user = df.groupby('userId')['rating'].count()

    ratings_per_movie_df = pd.DataFrame(ratings_per_movie)
    ratings_per_user_df = pd.DataFrame(ratings_per_user)

    filtered_ratings_per_movie_df = ratings_per_movie_df[ratings_per_movie_df.rating >= 300].index.tolist()
    filtered_ratings_per_user_df = ratings_per_user_df[ratings_per_user_df.rating >= 1500].index.tolist()
    
    df = df[df.movieId.isin(filtered_ratings_per_movie_df)]
    df = df[df.userId.isin(filtered_ratings_per_user_df)]
    return df