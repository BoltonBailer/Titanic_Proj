import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

if __name__ == '__main__':
    output = r'C:\Users\devlu\Downloads\DPL\DPL\labs\output\week2_output'
    path = r'C:\Users\devlu\Downloads\DPL\DPL\labs\data\tetuan-power.csv'
    seed = 123857

    df = pd.read_csv(path)

    df.info()

    #the data is complete, no nulls, the data is in floats, no weird symbols or commas.
    #input = Temperature,Humidity,Wind Speed,general diffuse flows,diffuse flows
    #output = Zone 1 Power Consumption

    #drop no numeric
    df = df.drop(["DateTime","Zone 2  Power Consumption","Zone 3  Power Consumption"], axis=1)

    in_feat = (["Temperature","Humidity","Wind Speed","general diffuse flows","diffuse flows"])
    out_feat = (["Zone 1 Power Consumption"])

    X = df[in_feat]
    y = df[out_feat]

    #TODO:Corr plot
    plt.figure()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Corr map')
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'corr_heatmap.png'))
    plt.close()

    #TODO:Collinearity plot
    plt.figure()
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
    plt.title('collinear')
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'collinearity_plot.png'))
    plt.close()

    #TODO:Draw a histogram of output feat
    plt.figure()
    sns.histplot(y)
    plt.title('Zone 1 Power Consumption')
    plt.xlabel('Zone 1 Power Consumption')
    plt.ylabel('Freq')
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'erp_histo.png'))
    plt.close()
