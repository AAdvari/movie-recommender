import mlflow
from movie_recommender import RecommenderSystemModel

model_path = "recommender-model"

mlflow.start_run()
mlflow.pyfunc.log_model(model_path, python_model=RecommenderSystemModel())
mlflow.end_run()