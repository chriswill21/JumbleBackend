import numpy as np
from random import shuffle
import recommender_system
from collections import deque

# get dictionary of all jobs and map them to index
data = []
category_to_id = {'accountancy qualified jobs': 0, 'finance jobs': 1, 'factory jobs': 2, 'it jobs': 3, 'purchasing jobs': 4, 'graduate training internships jobs': 5,
                  'engineering jobs': 6, 'catering jobs': 7, 'general insurance jobs': 8, 'energy jobs': 9, 'logistics jobs': 10, 'social care jobs': 11,
                  'motoring automotive jobs': 12, 'education jobs': 13, 'leisure tourism jobs': 14, 'admin secretarial pa jobs': 15, 'security safety jobs': 16,
                  'marketing jobs': 17, 'recruitment consultancy jobs': 18, 'retail jobs': 19, 'estate agent jobs': 20, 'health jobs': 21, 'hr jobs': 22,
                  'construction property jobs': 23, 'customer service jobs': 24, 'accountancy jobs': 25, 'charity jobs': 26, 'law jobs': 27, 'other jobs': 28,
                  'strategy consultancy jobs': 29, 'sales jobs': 30, 'banking jobs': 31, 'science jobs': 32, 'media digital creative jobs': 33, 'fmcg jobs': 34, 'training jobs': 35, 'apprenticeships jobs': 36}


category_to_id = {'accountancy qualified jobs': 0, 'accountancy jobs': 1, 'banking jobs': 2, 'finance jobs': 3, 'purchasing jobs': 4, 'sales jobs': 5,
                  'marketing jobs': 6, 'retail jobs': 7, 'fmcg jobs': 8, 'catering jobs': 9, 'social care jobs': 10, 'charity jobs': 11,
                  'leisure tourism jobs': 12, 'education jobs': 13, 'admin secretarial pa jobs': 14, 'graduate training internships jobs': 15, 'training jobs': 16,
                  'media digital creative jobs': 17, 'apprenticeships jobs': 18, 'security safety jobs': 19, 'construction property jobs': 20, 'motoring automotive jobs': 21,'factory jobs': 22, 'science jobs': 23,
                  'energy job': 24, 'health jobs': 25,'engineering jobs': 26, 'it jobs': 27, 'logistics jobs': 28, 'strategy consultancy jobs': 29,
                  'law jobs': 30, 'hr jobs': 31, 'general insurance jobs': 32, 'estate agent jobs': 33, 'recruitment consultancy jobs': 34, 'customer service jobs': 35, 'other jobs': 36}

# make multiple distirbutions over indices to represent how much different people would prefer different jobs, each distribution is a person
# first make list out if indeces, shuffle, then make normal distribution
# 100 distributions
# 30,000 users


dict = {}

def make_distribution():
    val = np.round(np.random.normal(18, 6, 1000))
    for i in val:
        if i not in dict:
            dict[i] = 1
        else:
            dict[i] += 1

    distribution = []

    for j in dict:
        distribution.append(dict[j]/len(val))

    print(val[:100])
    val.sort()
    print(val)
    distribution.sort()
    distribution.reverse()
    print(distribution)

    sums = 0
    total = []
    for i in range(0, len(distribution)-2,2):
        total.append(distribution[i])
        sums += distribution[i]

    total.sort()
    total.reverse()
    total = total[:18]
    print(sums)
    a = total.copy()
    total.reverse()
    b = total.copy()
    p = b + a
#
# print(p)
# # p = [0.001, 0.005, 0.006, 0.008, 0.022, 0.026, 0.041, 0.049, 0.058, 0.071, 0.072, 0.081, 0.087, 0.087, 0.081, 0.072, 0.071, 0.058, 0.049, 0.041, 0.026, 0.022, 0.008, 0.006, 0.005, 0.001]
# p = [0.001, 0.001, 0.002, 0.005, 0.007, 0.009, 0.011, 0.011, 0.022, 0.027, 0.03, 0.045, 0.048, 0.05, 0.054, 0.058,
#          0.064, 0.074, .08, 0.074, 0.064, 0.058, 0.054, 0.05, 0.048, 0.045, 0.03, 0.027, 0.022, 0.011, 0.011, 0.009,
#          0.007, 0.005, 0.002, 0.001, 0.001]
# print(.1179999999999992/37)
# t = 0
# for i in range(len(p)):
#     p[i] -= 0.003945945945945938
#     if p[i] < 0:
#         p[i] = 0
#     t += p[i]
#
# print(p)
# print(t)

p = [0, 0, 0, 0.001054054054054062, 0.003054054054054062, 0.005054054054054061, 0.007054054054054061,
         0.007054054054054061, 0.01805405405405406, 0.02305405405405406, 0.02605405405405406, 0.04105405405405406,
         0.04405405405405406, 0.046054054054054064, 0.05005405405405406, 0.054054054054054064, 0.06005405405405406,
         0.07005405405405406, 0.08837837837837857, 0.07005405405405406, 0.06005405405405406, 0.054054054054054064,
         0.05005405405405406, 0.046054054054054064, 0.04405405405405406, 0.04105405405405406, 0.02605405405405406,
         0.02305405405405406, 0.01805405405405406, 0.007054054054054061, 0.007054054054054061, 0.005054054054054061,
         0.003054054054054062, 0.001054054054054062, 0, 0, 0]


import pickle
def gaussian_to_distributions():
    distributions = []
    gaussian = deque(p)
    for i in range(37):
        distributions.append(list(gaussian))
        gaussian.rotate(1)

    pickle.dump(distributions, open("distributions", "wb"))
    return distributions


def get_data(data_type):
    x = pickle.load(open(data_type, "rb"))
    return x

user_count = 0

def iterations_for_making_dataset(user_mult, distributions):
    global user_count
    index_list = [i for i in range(37)]
    # shuffle(index_list)

    for i in range(100):
        selected_distribution_index = np.random.choice(index_list)
        selected_distribution = distributions[selected_distribution_index]
        val = np.random.choice(index_list, p = selected_distribution)
        if val not in dict:
            dict[val] = 1
        else:
            dict[val] += 1

    result = []

    for i in range(100):

        for key in dict.keys():

            entry = (user_mult + i, key, dict[key])
            if entry[0] > user_count:
                user_count = entry[0]
            result.append(entry)
        if i == 0 and user_mult == 0:
            print(result)

    return result

def make_dataset(new_user_data, distributions):
    global user_count
    train_data = []
    validation_data = []
    # iterate users
    for i in range(300):
        # got back is 100 users of the same distribution
        got_back = iterations_for_making_dataset(100 * i, distributions)
        shuffle(got_back)

        # remove data (equivalent of a cell in the UxV matrix) and add to use for validation data

        validation_data.append(got_back.pop(0))
        train_data += got_back
        # validation_data.append(got_back[0])
    for data_point in new_user_data:
        new_user_category = data_point[0]
        new_user_rating = data_point[1]

        train_data.append((user_count+1,new_user_category, new_user_rating))
    pickle.dump(train_data, open("train_data", "wb"))
    pickle.dump(validation_data, open("validation_data", "wb"))

def train_model(new_user_data, distributions):
    global user_count
    make_dataset(new_user_data, distributions)
    train_data = get_data("train_data")
    validation_data = get_data("validation_data")

    # f = open("model_result_details.txt", "w+")
    # print(recommender_system.mf_als(train_data, validation_data,lam=1, max_iter=30, k=20), file=f) #lam=0.7, max_iter=200, k=15), file=f)
    recommender_system.compute_and_save_large_model(train_data, validation_data)
    # f.close()



def test_model():
    global user_count
    x = recommender_system.load_model()

    data_validate = []
    for i in range(300):
        got_back = make_dataset(100*i)
        data_validate += got_back


    f = open("model_validation_results.txt", "w+")
    print('validate rmse=', recommender_system.rmse(data_validate, x), file=f)
    # print(recommender_system.compute_and_save_large_model(train_data, validation_data), file=f)

    f.close()

def get_rankings():
    global user_count
    #need to keep track of the row that the user was added to for the data
    # then use this and for this row in the U matrix of the model and for every job, compute the rating
    # map this rating to the job, sort them, and return the list of sorted job categories
    model = recommender_system.load_model()
    ranking_dict = {}
    data = [(0, 26, 5), (0, 9, 6), (0, 28, 3), (0, 13, 6), (0, 30, 12), (0, 3, 2), (0, 6, 11), (0, 2, 9), (0, 4, 6), (0, 36, 1), (0, 12, 4), (0, 33, 4), (0, 31, 5), (0, 17, 5), (0, 27, 3), (0, 16, 3), (0, 21, 4), (0, 10, 1), (0, 1, 4), (0, 35, 4), (0, 14, 1), (0, 23, 1)]

    for i in range(37):
        ranking_dict[i] = recommender_system.pred((user_count+1, i, None), model)

    rankings = sorted(ranking_dict, key=lambda k: ranking_dict[k])

    return rankings




# gaussian_to_distributions()
# distributions = get_data("distributions")
# train_model()
#self.response.write(train_model())
# test_model()
# single user: [(0, 26, 5), (0, 9, 6), (0, 28, 3), (0, 13, 6), (0, 30, 12), (0, 3, 2), (0, 6, 11), (0, 2, 9), (0, 4, 6), (0, 36, 1), (0, 12, 4), (0, 33, 4), (0, 31, 5), (0, 17, 5), (0, 27, 3), (0, 16, 3), (0, 21, 4), (0, 10, 1), (0, 1, 4), (0, 35, 4), (0, 14, 1), (0, 23, 1)]



distributions = None
def main(new_user):
    global user_count
    user_count = 0
    distributions = get_data("distributions")
    train_model(new_user, distributions)
    return get_rankings()


if __name__ == "__main__":
    main()