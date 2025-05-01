import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix



if __name__ == '__main__':

    data_dir = './data'
    out_dir = './output'
    filename = 'titanic_passengers.csv'
    path = os.path.join(data_dir,filename)
    seed = 42042

    df = pd.read_csv(path)

    #df.info()

    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    #hitogram stuff
    plt.figure()
    sns.histplot(df['Age'], bins=30)
    plt.title('Age distro')
    file_name = 'Titanic_age_histo.png'
    path = os.path.join(out_dir, file_name)
    plt.savefig(path)
    plt.close()

    #corr metrix
    plt.figure()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    file_name = 'Titanic_age_corr.png'
    path = os.path.join(out_dir, file_name)
    plt.savefig(path)
    plt.close()
    
    #grouping features
    df['FamSize'] = df['SibSp'] + df['Parch']

    X = df.drop('Survived', axis=1)
    y = df['Survived']
    #train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    #validation step
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    #print
    print(f"{acc:.4f}")
    ConfusionMetrix = confusion_matrix(y_test, y_pred)
    print(ConfusionMetrix)

    # heatmap 2
    plt.figure()
    sns.heatmap(ConfusionMetrix, annot=True, fmt='d', cmap='Blues')
    plt.title('confusin metrix')
    file_name = 'Titanic_logistic_Matrix.png'
    path = os.path.join(out_dir, file_name)
    plt.savefig(path)
    plt.close()



    
