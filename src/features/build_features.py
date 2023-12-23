import pandas as pd
import numpy as np
import pathlib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sys

#Load Dataset
def load_dataset(path):
    data = pd.read_csv(path)

    # Make a individual column
    data["individual_fare"] = data["Fare"]/(data["Parch"]+data["SibSp"]+1)

    # Make a Total member
    data["total_member"]=data["SibSp"]+data["Parch"]

    # Make a Total member to a family size column
    def family(data):
        if data == 1:
            return 'alone'
        elif data>1 and data <5:
            return "small"
        else:
            return "large"

    data["family_size"]=data["total_member"].apply(family)   

    # Cabin
    data["Cabin"]=data["Cabin"].str[0] 

    # Drop column
    data.drop(columns=["PassengerId","Name","SibSp","Parch","total_member","Ticket","Fare"],inplace = True)

    return data


# built feature and do column transform
def make_feature(data):
    

    # Make a pipeline
    numeric_features = ['Pclass', 'Age', "individual_fare"]
    categorical_features = ['Sex', 'Cabin', 'Embarked', 'family_size']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Use mean imputation for numeric features
       
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Use most frequent imputation for categorical features
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    pipe = make_pipeline(preprocessor)

    data = pipe.fit_transform(data)

    data = pd.DataFrame(data)

    return data

    

#save data

def save_data(data,outputpath):
    pathlib.Path(outputpath).mkdir(parents=True, exist_ok=True)
    data.to_csv(outputpath + "processed.csv",index=False)



 
def main():
    # Current path
    curr = pathlib.Path(__file__)

    # Home path
    home_dir = curr.parent.parent.parent
    
    input_file = sys.argv[1]

    # data path
    data_path = home_dir.as_posix() + input_file

    # output path
    outputpath = home_dir.as_posix() + '/data/build_data/'

    # dataset
    df = load_dataset(data_path)

    # make feature
    data = make_feature(df)

    # save data
    save_data(data,outputpath)


if __name__ == "__main__":
     main()    