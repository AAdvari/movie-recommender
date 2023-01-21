import warnings
import mlflow

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import mlflow
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


#################### Content Base ####################
class ContentBasedRecommendation():
    def __init__(self):
        self.credits_ds, self.links_ds, self.ratings_ds, self.keywords_ds, self.movies_metadata_ds = self.load_datasets()
        self.df = self.process_datasets(self.movies_metadata_ds, self.credits_ds, self.keywords_ds)
        self.vectorized_data = self.vectorize_data(self.df)
        self.similarity = self.calculate_similarity(self.vectorized_data)

    def load_datasets(self):
        credits_ds = pd.read_csv('./dataset/IMDB/credits.csv')
        links_ds = pd.read_csv('./dataset/IMDB/links_small.csv')
        ratings_ds = pd.read_csv('./dataset/IMDB/ratings_small.csv')
        keywords_ds = pd.read_csv('./dataset/IMDB/keywords.csv')
        movies_metadata_ds = pd.read_csv('./dataset/IMDB/movies_metadata.csv')
        return credits_ds, links_ds, ratings_ds, keywords_ds, movies_metadata_ds

    def process_movies_metadata(self, movies_metadata_ds):
        mlflow.log_param("Movies_Metadata Shape Before Data Cleaning", movies_metadata_ds.shape)
        mlflow.log_param("Movies_Metadata Column Before Cleaning", movies_metadata_ds.columns)
        movies_metadata_ds = movies_metadata_ds[movies_metadata_ds['status'] == 'Released']
        movies_metadata_ds = movies_metadata_ds[movies_metadata_ds['vote_count'] > 40]
        movies_metadata_ds = movies_metadata_ds[movies_metadata_ds['vote_average'] >= 5]
        important_col = ['id', 'genres', 'overview', 'original_title', 'belongs_to_collection']
        movies_metadata_ds = movies_metadata_ds[important_col]
        movies_metadata_ds.reset_index(inplace=True, drop=True)
        movies_metadata_ds['genres'] = movies_metadata_ds['genres'].apply(
            lambda x: ' '.join([i['name'].lower().replace(' ', '') for i in eval(x)]))
        movies_metadata_ds['belongs_to_collection'] = movies_metadata_ds['belongs_to_collection'].apply(
            lambda x: eval(str(x))['name'].lower().replace(' ', '') if str(x).lower() != 'nan' else '')
        movies_metadata_ds = movies_metadata_ds[movies_metadata_ds['id'].str.isnumeric()]
        movies_metadata_ds['id'] = movies_metadata_ds['id'].astype(int)
        mlflow.log_param("Movies_Metadata Shape After Data Cleaning", movies_metadata_ds.shape)
        mlflow.log_param("Movies_Metadata Column After Cleaning", movies_metadata_ds.columns)
        return movies_metadata_ds

    def process_credits(self, credits_ds):
        mlflow.log_param("Credits-Dataset Shape Before Data Cleaning", credits_ds.shape)
        mlflow.log_param("Credits-Dataset Columns Before Cleaning", credits_ds.columns)
        credits_ds['cast'] = credits_ds['cast'].apply(
            lambda x: ' '.join([i['name'].lower().replace(' ', '') for i in eval(x)]))

        credits_ds['crew'] = credits_ds['crew'].apply(
            lambda x: [i['name'].lower().replace(' ', '') if i['job'] == 'Director' else '' for i in eval(x)])
        credits_ds['crew'] = credits_ds['crew'].apply(lambda x: ' '.join([i for i in x if i != '']))
        credits_ds['cast'] = credits_ds.apply(lambda x: x.loc['cast'] + ' ' + x.loc['crew'], axis=1)
        credits_ds = credits_ds[['id', 'cast']]
        credits_ds.reset_index(inplace=True, drop=True)
        mlflow.log_param("Credits-Dataset Shape after Data Cleaning", credits_ds.shape)
        mlflow.log_param("Credits-Dataset Columns after Cleaning", credits_ds.columns)
        return credits_ds

    def process_keywords(self, keywords_ds):
        keywords_ds['keywords'] = keywords_ds['keywords'].apply(
            lambda x: ' '.join([i['name'].lower().replace(' ', '') for i in eval(x)]))
        return keywords_ds

    def make_general_df(self, movies_metadata_ds, credits_ds, keywords_ds):
        df = pd.merge(movies_metadata_ds, keywords_ds, on='id', how='left')
        df = pd.merge(df, credits_ds, on='id', how='left')
        df.reset_index(inplace=True)
        df.drop(columns=['index'], inplace=True)
        return df

    def clean_general_df(self, df):
        col = list(df.columns)
        col.remove('id')
        col.remove('genres')
        col.remove('original_title')
        df['title'] = df['original_title']
        df['token'] = df['genres']
        for i in col:
            df['token'] = df['token'] + ' ' + df[i]
        df = df[['id', 'title', 'token']]
        df.drop(df[df['token'].isnull()].index, inplace=True)
        mlflow.log_param("Merged Dataset Shape", df.shape)
        mlflow.log_param("Merged Dataset Columns", df.columns)
        return df

    def process_datasets(self, movies_metadata_ds, credits_ds, keywords_ds):
        movies_metadata_ds = self.process_movies_metadata(movies_metadata_ds)
        credits_ds = self.process_credits(credits_ds)
        keywords_ds = self.process_keywords(keywords_ds)
        df = self.make_general_df(movies_metadata_ds, credits_ds, keywords_ds)
        df = self.clean_general_df(df)
        return df

    def vectorize_data(self, df, MAX_FEATURES=5000):
        mlflow.log_metric("MAX_FEATURES in vectorizing tags column", MAX_FEATURES)
        tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
        vectorized_data = tfidf.fit_transform(df['token'].values)
        return vectorized_data

    def calculate_similarity(self, vectorized_data):
        similarity = cosine_similarity(vectorized_data)
        mlflow.log_param("Movies-Similarity", similarity)
        return similarity

    def content_recommendation_by_movie(self, df, similarity, title, number=20):
        if len(df[df['title'] == title]) == 0:
            return []
        movie_id = df[df['title'] == title].index[0]
        distances = similarity[movie_id]

        fig, ax = plt.subplots()
        ax.plot(sorted(distances[:number], reverse=True))
        plt.title("similarities")
        plt.savefig("similarities.png")
        mlflow.log_figure(fig, "similarities.png")
        plt.close()

        movies = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
        return [df.iloc[i[0]].title for i in movies[:number]]

    def content_recommendation_by_user(self, df, ratings_ds, similarity, user_id, number=20):
        user_rate_ds = ratings_ds[ratings_ds['userId'] == user_id]
        sort = user_rate_ds.sort_values(by='rating', ascending=False)
        movie_id = sort['movieId']
        movie_list = [df[df['id'] == id]['title'].values[0] for id in movie_id if len(df[df['id'] == id]['title']) > 0]
        result = [self.content_recommendation_by_movie(df, similarity, str(title)) for title in movie_list]
        return list(itertools.chain.from_iterable(result))

    def predict_by_movie(self, title):
        recommendations = self.content_recommendation_by_movie(self.df, self.similarity, title)
        return recommendations

    def predict(self, user_id):
        recommendations = self.content_recommendation_by_user(self.df, self.ratings_ds, self.similarity, user_id)
        return recommendations

    def local_content_base_test(self):
        print(self.predict_by_movie('Toy Story'))
        print('***************************************************************************')
        print(self.predict_by_movie('Jumanji'))
        print('***************************************************************************')
        print(self.predict_by_movie('Rocky III'))
        print('***************************************************************************')
        print(self.predict(1))


#################### Collaborative ####################
class CollaborativeRecommendation():
    def __init__(self):
        self.movie_df, self.rate_df = self.load_dataframes()
        self.sparse_matrix = self.make_sparse_matrix(self.rate_df)
        self.similarities_sparse = self.make_similarity_sparse(self.sparse_matrix)

    def load_dataframes(self):
        movie_df = pd.read_csv('./dataset/IMDB/movies_metadata.csv')
        rate_df = pd.read_csv('./dataset/IMDB/ratings_small.csv')
        return movie_df, rate_df

    def make_sparse_matrix(self, rate_df):
        user_max = rate_df['userId'].max()
        movie_max = rate_df['movieId'].max()
        i = torch.LongTensor(rate_df[['userId', 'movieId']].to_numpy())
        v = torch.FloatTensor(rate_df[['rating']].to_numpy().flatten())
        sparse_matrix = torch.sparse.FloatTensor(i.t(), v, torch.Size([user_max + 1, movie_max + 1])).to_dense()
        return sparse_matrix

    def make_similarity_sparse(self, sparse_matrix):
        similarities_sparse = cosine_similarity(sparse_matrix, dense_output=False)
        mlflow.log_param("users similarity sparse matrix", str(similarities_sparse)[:400])
        return similarities_sparse

    def top_n_index_sparse(self, similarities_sparse, user_id, number=20):
        user_row = similarities_sparse[user_id]

        fig, ax = plt.subplots()
        ax.plot(list(sorted(user_row, reverse=True))[:number])
        plt.title("users-similarities")
        plt.savefig("users-similarities.png")
        mlflow.log_figure(fig, "users-similarities.png")
        plt.close()

        user_details = list(map(lambda x: (x[0], x[1]), enumerate(user_row)))
        sort = list(sorted(user_details, key=lambda x: x[1], reverse=True))
        # removing user itself
        sort = sort[1:]
        result = list(map(lambda x: x[0], sort[:number]))
        return result

    def user_top_movies(self, rate_df, user_id, number=10):
        user_rate = rate_df[rate_df['userId'] == user_id]
        sort = user_rate.sort_values(by='rating', ascending=False)
        number = number if number <= len(sort) else len(sort)
        result = sort['movieId'].values[:number]
        return result

    def recommendation_for_user(self, movie_df, rate_df, similarities_sparse, user_id, number=20):
        similar_users = self.top_n_index_sparse(similarities_sparse, user_id)
        movies = []
        for i in similar_users:
            similar_user_movies = self.user_top_movies(rate_df, i)
            [movies.append(j) for j in similar_user_movies]
        temp = rate_df[rate_df['userId'] == user_id]
        for i in movies:
            if len(temp[temp['movieId'] == i]) > 0:
                movies.remove(i)
        titles = [movie_df[movie_df['id'] == str(id)]['title'].values[0] for id in movies if
                  len(movie_df[movie_df['id'] == str(id)]['title']) > 0]
        number = number if number < len(titles) else len(titles)
        return titles[:number]

    def predict(self, user_id):
        recommendations = self.recommendation_for_user(self.movie_df, self.rate_df, self.similarities_sparse, user_id)
        return recommendations

    def local_collaborative_test(self):
        print(self.recommendation_for_user(self.movie_df, self.rate_df, self.similarities_sparse, 1))


#################### Ensemble ####################
class EnsembleRecommendation():
    def __init__(self):
        mlflow.end_run()
        mlflow.start_run()
        self.collab = CollaborativeRecommendation()
        self.content = ContentBasedRecommendation()

    def ensemble_recommendation_intersection_based(self, user_id, number=10):
        collaborative = self.collab.predict(user_id)
        content_based = self.content.predict(user_id)
        result = list(set(collaborative) & set(content_based))  # finding intersect
        for i in result:
            collaborative.remove(i)
            content_based.remove(i)
        collaborative_index = 0
        content_base_index = 0
        while len(result) < number:
            if collaborative_index > content_base_index:
                result.append(content_based[content_base_index])
                content_base_index = content_base_index + 1
            else:
                result.append(collaborative[collaborative_index])
                collaborative_index = collaborative_index + 1
        return result

    def ensemble_recommendation_collaborative_based(self, user_id, number=10):
        collaborative = self.collab.predict(user_id)
        results = []
        for movie in collaborative:
            recommended_movies = self.content.predict_by_movie(movie)
            for i in recommended_movies:
                results.append(i) if i not in results else None
        return results[:number]

    def predict(self, user_id, intersection_base=True):
        if intersection_base:
            return self.ensemble_recommendation_intersection_based(user_id, number=10)
        else:
            return self.ensemble_recommendation_collaborative_based(user_id, number=10)

    def local_test(self):
        print(self.predict(1, intersection_base=True))
        print('***************************************************************************')
        print(self.predict(1, intersection_base=False))


#################### Testing Models ####################
# warnings.filterwarnings("ignore")
# ensemble = EnsembleRecommendation()
# ensemble.content.local_content_base_test()
# print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
# ensemble.collab.local_collaborative_test()
# print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
# ensemble.local_test()
#################### MLOps Model ####################
class RecommenderSystemModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.ensemble = EnsembleRecommendation()
        self.collab = self.ensemble.collab
        self.content = self.ensemble.content

    def add_data(self, df):
        pre_df = pd.read_csv('./dataset/IMDB/ratings_small.csv')
        new_df = pd.concat([pre_df, df])
        new_df.to_csv('./dataset/IMDB/ratings_small.csv', index=False)

        movie_df = pd.read_csv('./dataset/IMDB/movies_metadata.csv')
        movies = list(df['movieId'].unique())
        movies = list(map(lambda x: str(x), movies))

        def vote_update(row):
            if row['id'] not in movies:
                return row
            vote = df[df['movieId'] == int(row['id'])]['rating'].values.sum()
            count = len(df[df['movieId'] == int(row['id'])])
            pre_count = int(row['vote_count'])
            pre_avg = float(row['vote_average'])
            new_avg = ((pre_avg * pre_count) + vote) / (count + pre_count)
            row['vote_average'] = str(new_avg)
            row['vote_count'] = str(pre_count + count)
            return row

        movie_df[['vote_count', 'vote_average']] = \
            movie_df.apply(lambda x: vote_update(x), axis=1)[['vote_count', 'vote_average']]
        movie_df.to_csv('./dataset/IMDB/movies_metadata.csv', index=False)
        self.ensemble = EnsembleRecommendation()
        self.collab = self.ensemble.collab
        self.content = self.ensemble.content

    def predict(self, context, model_input):
        if type(model_input) == list:
            return self.my_custom_function(model_input)
        else:
            return self.add_data(model_input)

    def my_custom_function(self, model_input):
        user_id = model_input[0]
        type = model_input[1]
        if type == 1:
            return self.content.predict(user_id)
        if type == 2:
            return self.collab.predict(user_id)
        if type == 3:
            return self.ensemble.predict(user_id, intersection_base=False)
        if type == 4:
            return self.ensemble.predict(user_id, intersection_base=True)
        return 0


# warnings.filterwarnings("ignore")
# x = pd.read_csv('./dataset/IMDB/movies_metadata.csv')
# print(x[x['id'] == '862'].to_string())

EnsembleRecommendation()

# rs = RecommenderSystemModel()
# rs.add_data(pd.DataFrame(
#     {
#         'userId': [1, 1, 1],
#         'movieId': [862, 862, 862],
#         'rating': [1000, 1000, 1000]
#     }
# ))
# x = pd.read_csv('./dataset/IMDB/movies_metadata.csv')
# print(x[x['id'] == '862'].to_string())
# x = pd.read_csv('./dataset/IMDB/ratings_small.csv')
# print(x[x['movieId'] == 862].to_string())

# sending request command
## mlflow models serve -m recommender-model -p 6000
# curl http://127.0.0.1:6000/invocations -H 'Content-Type: application/json' -d '{
#   "inputs": [1,1]
# }'

# curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
# "dataframe_records": [
#     {"userId": 1,"movieId": 2,"rating": 3},
#     {"userId": 4,"movieId": 5,"rating": 6}
# ]
# }'