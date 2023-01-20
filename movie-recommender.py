#!/usr/bin/env python
# coding: utf-8

# In[121]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import mlflow
import matplotlib.pyplot as plt

# # Loading dataset's

# In[4]:


credits_ds = pd.read_csv('./dataset/IMDB/credits.csv')
credits_ds

# In[5]:


links_ds = pd.read_csv('./dataset/IMDB/links_small.csv')
links_ds

# In[6]:


ratings_ds = pd.read_csv('./dataset/IMDB/ratings_small.csv')
ratings_ds

# In[7]:


keywords_ds = pd.read_csv('./dataset/IMDB/keywords.csv')
keywords_ds

# In[8]:


movies_metadata_ds = pd.read_csv('./dataset/IMDB/movies_metadata.csv')
movies_metadata_ds

# In[9]:

with mlflow.start_run():
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

mlflow.log_param("Movies_Metadata Shape After Data Cleaning", movies_metadata_ds.shape)
mlflow.log_param("Movies_Metadata Column After Cleaning", movies_metadata_ds.columns)

movies_metadata_ds

# In[10]:


mlflow.log_param("Credits-Dataset Shape Before Data Cleaning", credits_ds.shape)
mlflow.log_param("Credits-Dataset Columns Before Cleaning", credits_ds.columns)

credits_ds['cast'] = credits_ds['cast'].apply(lambda x: ' '.join([i['name'].lower().replace(' ', '') for i in eval(x)]))

credits_ds['crew'] = credits_ds['crew'].apply(
    lambda x: [i['name'].lower().replace(' ', '') if i['job'] == 'Director' else '' for i in eval(x)])
credits_ds['crew'] = credits_ds['crew'].apply(lambda x: ' '.join([i for i in x if i != '']))
credits_ds['cast'] = credits_ds.apply(lambda x: x.loc['cast'] + ' ' + x.loc['crew'], axis=1)
credits_ds = credits_ds[['id', 'cast']]
credits_ds.reset_index(inplace=True, drop=True)

mlflow.log_param("Credits-Dataset Shape after Data Cleaning", credits_ds.shape)
mlflow.log_param("Credits-Dataset Columns after Cleaning", credits_ds.columns)

credits_ds

# In[11]:


keywords_ds['keywords'] = keywords_ds['keywords'].apply(
    lambda x: ' '.join([i['name'].lower().replace(' ', '') for i in eval(x)]))
keywords_ds

# In[12]:


movies_metadata_ds = movies_metadata_ds[movies_metadata_ds['id'].str.isnumeric()]
movies_metadata_ds['id'] = movies_metadata_ds['id'].astype(int)

df = pd.merge(movies_metadata_ds, keywords_ds, on='id', how='left')
df = pd.merge(df, credits_ds, on='id', how='left')
df.reset_index(inplace=True)
df.drop(columns=['index'], inplace=True)
df

# In[13]:


col = list(df.columns)
col.remove('id')
col.remove('genres')
col.remove('original_title')
df['title'] = df['original_title']
df['token'] = df['genres']
for i in col:
    df['token'] = df['token'] + ' ' + df[i]
df = df[['id', 'title', 'token']]
df

# In[14]:


df.drop(df[df['token'].isnull()].index, inplace=True)

mlflow.log_param("Merged Dataset Shape", df.shape)
mlflow.log_param("Merged Dataset Columns", df.columns)

df

# In[15]:

MAX_FEATURES = 5000
mlflow.log_metric("MAX_FEATURES in vectorizing tags column", MAX_FEATURES)

tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
vectorized_data = tfidf.fit_transform(df['token'].values)
vectorized_data

# In[16]:
similarity = cosine_similarity(vectorized_data)
mlflow.log_param("Movies-Similarity", similarity)


# In[17]:
def recommend_by_movie(title, number=20):
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


print(recommend_by_movie('Toy Story', 10))

print('***************************************************************************')

print(recommend_by_movie('Jumanji', 10))

print('***************************************************************************')

print(recommend_by_movie('Rocky III', 10))


# In[30]:
def recommen_by_user(user_id, number=20):
    user_rate_ds = ratings_ds[ratings_ds['userId'] == user_id]
    sort = user_rate_ds.sort_values(by='rating', ascending=False)
    movie_id = sort['movieId']
    movie_list = [df[df['id'] == id]['title'].values[0] for id in movie_id if len(df[df['id'] == id]['title']) > 0]
    result = [recommend_by_movie(str(title)) for title in movie_list]
    return list(itertools.chain.from_iterable(result))


recommen_by_user(1)

# # Colaborative

# In[19]:


import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

# In[20]:


movie_df = pd.read_csv('./dataset/IMDB/movies_metadata.csv')
movie_df

# In[21]:


rate_df = pd.read_csv('./dataset/IMDB/ratings_small.csv')
rate_df

# In[22]:


user_max = rate_df['userId'].max()
movie_max = rate_df['movieId'].max()

i = torch.LongTensor(rate_df[['userId', 'movieId']].to_numpy())
v = torch.FloatTensor(rate_df[['rating']].to_numpy().flatten())

sparse_matrix = torch.sparse.FloatTensor(i.t(), v, torch.Size([user_max + 1, movie_max + 1])).to_dense()

print(sparse_matrix.shape)
sparse_matrix

# In[23]:


similarities_sparse = cosine_similarity(sparse_matrix, dense_output=False)
print(similarities_sparse.shape)

mlflow.log_param("users similarity sparse matrix", similarities_sparse)


# In[24]:


def top_n_idx_sparse(user_id, number=20):
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
    return list(map(lambda x: x[0], sort[:number]))


user_user_similar = top_n_idx_sparse(1)
user_user_similar


# In[25]:


def user_top_movies(user_id, number=10):
    user_rate = rate_df[rate_df['userId'] == user_id]
    sort = user_rate.sort_values(by='rating', ascending=False)
    number = number if number <= len(sort) else len(sort)
    return sort['movieId'].values[:number]


user_325_top_movies = user_top_movies(325)
user_325_top_movies


# In[26]:


def recommendation_for_user(user_id, number=20):
    similar_users = top_n_idx_sparse(user_id)
    movies = []
    for i in similar_users:
        similar_user_movies = user_top_movies(i)
        [movies.append(j) for j in similar_user_movies]
    temp = rate_df[rate_df['userId'] == user_id]
    for i in movies:
        if len(temp[temp['movieId'] == i]) > 0:
            movies.remove(i)
    titles = [movie_df[movie_df['id'] == str(id)]['title'].values[0] for id in movies if
              len(movie_df[movie_df['id'] == str(id)]['title']) > 0]
    number = number if number < len(titles) else len(titles)
    return titles[:number]


recommendation_for_user(1)


# # Ensemble

# In[48]:


def ensemble_recommendation_intersection_based(user_id, number=10):
    collaborative = recommendation_for_user(user_id)
    content_based = recommen_by_user(user_id)
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


def ensemble_recommendation_collaborative_based(user_id, number=10):
    collaborative = recommendation_for_user(user_id)
    results = []
    for movie in collaborative:
        recommended_movies = recommend_by_movie(movie)
        for i in recommended_movies:
            results.append(i) if i not in results else None
    return results[:number]


def ensemble_recommendation(user_id, number=10, intersection_base=True):
    if intersection_base:
        return ensemble_recommendation_intersection_based(user_id, number)
    else:
        return ensemble_recommendation_collaborative_based(user_id, number=10)


# In[49]:


recommendation = ensemble_recommendation(1, intersection_base=True)
recommendation

# In[50]:


recommendation = ensemble_recommendation(1, intersection_base=False)
recommendation

# # MLOps

# In[ ]:
