import os
from flask import Flask, jsonify, request
from flask_mysqldb import MySQL
import uuid

from db_helper import initDb,create_job_record
from web_service_helper import initWebServices, upload_file
from predict import predict_image, initModel

app = Flask(__name__)
model = initModel()

initWebServices(app)
dbInstance = initDb(app)

@app.route('/')
def hello_world():
    # create_job_record(dbInstance, "Test note", "Test class", "test record")
    return jsonify({"status": 1})

@app.route('/predict', methods=['GET', 'POST'])
def predict_endpoint():
    if request.method == 'GET':
        return '''
            <h1>Upload new File</h1>
            <form method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="text" name="note">
            <input type="submit">
            </form>
            '''

    result = {"success": False, "msg": "", "predict": None, "data": None }
    upload = upload_file(app)
    note = request.form.get("note")

    if(upload.isUploaded != True):
        result['msg'] = upload.msg
        return jsonify(result)
    
    predict_result = predict_image(model, upload.get_uploaded_filename())
    jobId = uuid.uuid4()
    create_job_record(dbInstance, jobId, note, predict_result, upload.get_uploaded_filename())
    # print("[predict] -> img" + upload.get_uploaded_filename())
    # print("[predict] -> result " + predict_result)
    result['predict'] = predict_result
    # result['predict'] = predict_result
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)