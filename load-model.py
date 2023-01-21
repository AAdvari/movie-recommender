import mlflow

model_path = "recommender-model"

loaded_model = mlflow.pyfunc.load_model(model_path)
print(loaded_model.predict([1, 1]))
print(loaded_model.predict([1, 2]))
print(loaded_model.predict([1, 3]))
print(loaded_model.predict([1, 4]))