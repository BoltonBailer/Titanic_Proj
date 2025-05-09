import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

if __name__ == '__main__':
    data = './data'
    out = './output'
    file_name = 'machine-data.csv'
    path = os.path.join(data, file_name)
    seed = 123857

    df = pd.read_csv(path)
    
    # drop nulls
    df.dropna(inplace=True)

    # Only numbers
    numeric_df = df.select_dtypes(include='number')

    # set erp to three diff labels 
    df['erp_class'] = pd.qcut(df['erp'], q=3, labels=['low', 'medium', 'high'])

    # input/output

    in_feat = numeric_df.columns.drop('erp')
    out_feat = 'erp_class'

    X = df[in_feat]
    y = df[out_feat]

    model = Perceptron()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    model.fit(X_train, y_train)

    # Histogram
    plt.figure()
    sns.histplot(df['erp'], bins=30)
    plt.title('erp histo')
    plt.xlabel('erp')
    plt.ylabel('Freq')
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'erp_histo.png'))
    plt.close()

    # Heatmap
    plt.figure()
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Corr map')
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'corr_heatmap.png'))
    plt.close()

    # Collinearity
    plt.figure()
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
    plt.title('collinear')
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'collinearity_plot.png'))
    plt.close()

    # eval model
    preds = model.predict(X_test)
    print(accuracy_score(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
