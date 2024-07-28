import os
from flask import Flask, jsonify, request, Response
from flask_mysqldb import MySQL
from flask_cors import CORS, cross_origin
import uuid
import json

from db_helper import JobRecord, initDb,create_job_record, get_job_history
from db_helper import UserRecord, create_user, get_user_id, check_for_user_id, check_for_user_email
from web_service_helper import initWebServices, upload_file
from predict import predict_image, initModel

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = initModel()

initWebServices(app)
dbInstance = initDb(app)

@app.route('/')
def hello_world():
    # create_job_record(dbInstance, "Test note", "Test class", "test record")
    return jsonify({"status": 1})

@app.route('/user', methods=['POST'])
def user_signup_endpoint():
    result = {"success": False, "msg": "", "data": None}
    json = request.json

    # TODO trim and check values

    if not "firstName" in json:
        result['msg'] = "First Name is required"
        return result
    if not "lastName" in json:
        result['msg'] = "Last Name is required"
        return result
    if not "email" in json:
        result['msg'] = "Email is required"
        return result
    if not "password" in json:
        result['msg'] = "Password is required"
        return result
    
    firstName = json["firstName"]
    lastName = json["lastName"]
    email = json["email"]
    password = json["password"]

    user_exist = check_for_user_email(dbInstance, email)

    if (user_exist == True):
        result['msg'] = "User already exist"
        return result

    record = create_user(dbInstance, firstName, lastName, email, password)
    result['data'] = record.serialize()
    result["success"] = True
    return jsonify(result)

@app.route('/user/login', methods=['POST'])
def user_login_endpoint():
    result = {"success": False, "msg": "", "data": None}
    # TODO trim and check values
    json = request.json

    if not "name" in json:
        result['msg'] = "Username is required"
        return result
    if not "password" in json:
        result['msg'] = "Password is required"
        return result
    
    name = json["name"]
    password = json["password"]
    
    userId = get_user_id(dbInstance, name, password)
    result['data'] = userId
    result["success"] = userId != None and userId != ""
    return jsonify(result)

def validate_user(request, result)->Response:
    exists: bool = False

    if "auth" in request.headers:
        exists = check_for_user_id(dbInstance, request.headers['auth'])
    else:
        result['msg'] = "Authentication required to access the resource"
        return Response(json.dumps(result), status=401, mimetype='application/json')

    if not exists:
        result['msg'] = "Invalid authentication provided to access the resource"
        return Response(json.dumps(result), status=403, mimetype='application/json')
    
    return None #if everything is fine None will be the out put

@app.route('/test', methods=['POST'])
def check_user_auth_endpoint():
    result = {"success": False, "msg": "", "data": False}

    resp = validate_user(request, result)

    if not resp is None:
        return resp

    result["msg"] = "success"
    result["success"] = True
    return jsonify(result)
    

@app.route('/history', methods=['GET'])
def view_history_endpoint():
    result = {"success": False, "msg": "", "data": None }

    # user validation
    resp = validate_user(request, result)

    if not resp is None:
        return resp
    
    userId = request.headers['auth']
    records = get_job_history(dbInstance, userId)
    result['data'] = records
    
    return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    result = {"success": False, "msg": "", "data": None }

    # user validation
    resp = validate_user(request, result)

    if not resp is None:
        return resp
    
    userId = request.headers['auth']
    jobId = uuid.uuid4()

    # begin upload
    upload = upload_file(app, str(jobId))

    if(upload.isUploaded != True):
        result['msg'] = upload.msg
        return jsonify(result)
    
    # predic result
    predict_output = predict_image(model, upload.get_uploaded_filename())
    predict_result = predict_output[0]

    # save record
    create_job_record(dbInstance, jobId, predict_output[1], predict_result, upload.filename, userId)
    
    # TODO uncomment for debug trace
    # print("[predict] -> img" + upload.get_uploaded_filename())
    # print("[predict] -> result " + predict_result)
    
    result['success'] = True
    result['data'] = {
        "result": predict_result,
        "matches": str(predict_output[1])
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)