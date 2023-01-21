import mlflow
from movie_recommender import RecommenderSystemModel

model_path = "recommender-model"

mlflow.pyfunc.save_model(path=model_path, python_model=RecommenderSystemModel())