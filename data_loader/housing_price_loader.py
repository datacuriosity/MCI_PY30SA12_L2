from sklearn.datasets import load_boston, load_breast_cancer
import pandas as pd
# import streamlit as st


def load_boston_data():
    data = load_boston()
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['price'] = data['target']
    return df


def load_cancer_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['price'] = data['target']
    return df


if __name__ == '__main__':
    load_cancer_data()
#     df = load_boston_data()
#     st.header('This is housing price dataset')
#     df

