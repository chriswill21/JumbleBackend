from flask import Flask
import job_tagging_svm
app = Flask(__name__)


@app.route("/")
def hello():
    # return "Hello World!"
    a = job_tagging_svm.predict_job_category()
    print(a)
    return str(a)


if __name__ == "__main__":
    app.run()