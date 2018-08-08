import scipy
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from random import shuffle
from collections import deque
import pickle


def train_tagging_model():
    job_post_data_total = pd.read_csv("reed_uk.csv")
    job_post_data_trimmed = job_post_data_total.sample(frac=.5)
    # divide between training and test data
    max_num_words_in_list = 375172*1/4

    col = ['category', 'job_description']
    job_post_data = job_post_data_trimmed[col]
    job_post_data['category_id'], uniques = job_post_data['category'].factorize()


    posts = job_post_data["job_description"]
    tags = job_post_data["category"]


    train_size = int(len(posts) * .8)

    train_posts = posts[:train_size]
    train_tags = tags[:train_size]

    test_posts = posts[train_size:]
    test_tags = tags[train_size:]


    vocab_size = int(max_num_words_in_list)


    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

    # fit vectorizer on data set
    fitted_feature_vectorizer = tfidf.fit(posts)
    # transform dataset inputs to vectors
    transformed_feature_matrix = fitted_feature_vectorizer.transform(posts)
    features = transformed_feature_matrix.toarray()

    # saving the tokenizer
    with open('svm_tokenizer.pickle', 'wb') as handle:
        pickle.dump(fitted_feature_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    labels = tags
    features.shape

    job_post_data_trimmed.to_csv("data_halved", sep='\t')



    # MODEL SELECTION
    # models = [
    #     RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    #     LinearSVC(),
    #     LogisticRegression(random_state=0),
    # ]
    # CV = 5
    # cv_df = pd.DataFrame(index=range(CV * len(models)))
    # entries = []
    # for model in models:
    #   model_name = model.__class__.__name__
    #   accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    #   for fold_idx, accuracy in enumerate(accuracies):
    #     entries.append((model_name, fold_idx, accuracy))
    # cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    #
    # sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    # sns.stripplot(x='model_name', y='accuracy', data=cv_df,
    #               size=8, jitter=True, edgecolor="gray", linewidth=2)
    # plt.show()

    # ----------------------------------------------------------------------------------

    # MODEL EVALUATION
    model = LinearSVC()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, job_post_data.index, test_size=0.33, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model.score(X_test, y_test))
    print(X_train.shape)
    joblib.dump(model, 'svm_tagging_model.pkl')
    #
    # category_id_df = job_post_data[['category', 'category_id']].drop_duplicates().sort_values('category_id')
    # category_to_id = dict(category_id_df.values)
    # id_to_category = dict(category_id_df[['category', 'category_id']].values)
    #
    #
    # from sklearn.metrics import confusion_matrix
    # conf_mat = confusion_matrix(y_test, y_pred)
    # fig, ax = plt.subplots(figsize=(10,10))
    # sns.heatmap(conf_mat, annot=True, fmt='d',
    #             xticklabels=category_id_df.category.values, yticklabels=category_id_df.category.values)
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.show()

    # ---------------------------------------------------------------------------------


def predict_job_category():

    model = joblib.load("svm_tagging_model.pkl")

    job_post_data_total = pd.read_csv("reed_uk.csv")
    job_post_data_trimmed = job_post_data_total.sample(frac=.5)
    # divide between training and test data
    max_num_words_in_list = 375172 * 1 / 4

    col = ['category', 'job_description']
    job_post_data = job_post_data_trimmed[col]
    job_post_data['category_id'], uniques = job_post_data['category'].factorize()

    posts = job_post_data["job_description"]
    tags = job_post_data["category"]

    fitted_feature_matrix = joblib.load("svm_tokenizer.pickle")
    features = fitted_feature_matrix.transform(posts).toarray()
    labels = tags
    features.shape

    job_post_data_trimmed.to_csv("data_halved", sep='\t')

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, job_post_data.index, test_size=0.33, random_state=0)

    return model.score(X_test, y_test)

