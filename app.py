from flask import Flask, request, render_template
import job_tagging_svm
import compute_recommendation_model
app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def hello():
    return "Hello World!"
    a = job_tagging_svm.predict_job_category()
    print(a)
    return str(a)

@app.route("/tagging/<job_description>", methods=['GET', 'POST'])
def job_tagging(job_description):
    prediction = job_tagging_svm.predict_job_category(job_description)
    print(prediction)
    return str(prediction[0])

@app.route("/recommending/<new_user_data_string>", methods=['GET','POST'])
def job_recommendation(new_user_data_string):
    # new_user_data = list(new_user_data)
    new_user_data = []
    num_builder = ""
    data_point = []
    start = False
    list_data = list(new_user_data_string)
    list_data.pop(0)
    list_data.pop(-1)
    for i in list_data:
        if i == "[":
            start = True
        else:
            if start:
                if i != " " and i != "," and i != "]":
                    num_builder += i
                elif i == ",":
                    data_point.append(int(num_builder))
                    num_builder = ""
                elif i == "]":

                    data_point.append(int(num_builder))
                    new_user_data.append(tuple(data_point))
                    data_point = []
                    num_builder = ""
                    start = False


    prediction = compute_recommendation_model.main(new_user_data)
    return str(prediction)

if __name__ == "__main__":
    app.run()