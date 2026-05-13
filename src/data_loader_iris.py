from functools import lru_cache
import pandas as pd

### column names and class mappings
columns = ['Sepal Length','Sepal Width','Petal Length','Petal Width','Class']
classes = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica':2}

@lru_cache(maxsize=1)
def load_data(csv_path: str = "iris/iris.data") -> pd.DataFrame:
    print("Loading data from CSV")
    df = pd.read_csv(csv_path, names=columns, header = None)
    df['Class'] = df['Class'].map(classes)
    return df

