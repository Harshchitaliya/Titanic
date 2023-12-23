import pandas as pd
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
import yaml
import sys



def load_dataset(path):
    df = pd.read_csv(path)
    return df

def data_split(data,test_split,seed):
    X = data.drop(columns=["3"])
    y = data["3"]

    # train test split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_split,random_state=seed,stratify=y)
    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test,output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_path + 'X_train.csv', index=False)
    X_test.to_csv(output_path + 'X_test.csv', index=False)
    y_train.to_csv(output_path + 'y_train.csv', index=False)
    y_test.to_csv(output_path + 'y_test.csv', index=False)



def main():

    curr = pathlib.Path(__file__)
    home_dir = curr.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["make_dataset"]

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed/'

    # Load dataset
    df = load_dataset(data_path)

    # Train test split
    X_train, X_test, y_train, y_test = data_split(df,params["test_split"],params["seed"])

    #save a data
    save_data(X_train,X_test,y_train,y_test, output_path)

if __name__ == "__main__":
     main()
