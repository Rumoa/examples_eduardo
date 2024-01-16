import joblib
import numpy as np

array_to_save = np.linspace(0, 100)

joblib.dump(array_to_save, "file.job")


arra = joblib.load("file.job")
