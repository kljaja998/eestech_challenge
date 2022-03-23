from tpot import TPOTClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# pipeline_optimizer = TPOTClassifier()

df = pd.read_csv('./data/Train-dataset.csv')

MAPPING = {
    'Continental': 1,
    'Transitional': 2,
    'Marine': 3
}

df['D_Env'] = df['DEPOSITIONAL_ENVIRONMENT'].apply(lambda x: MAPPING[x])

Feature = df[['X', 'Y', 'MD', 'GR', 'RT', 'DEN', 'CN', 'D_Env']]

X = Feature

y = df['LITH_CODE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

pipeline_optimizer = TPOTClassifier(generations=100, population_size=100, cv=5, random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print("Finished fitting")
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')
