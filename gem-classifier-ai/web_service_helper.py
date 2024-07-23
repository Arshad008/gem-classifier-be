import os
from flask import Flask, flash, request, redirect, url_for

script_dir = os.path.dirname(__file__)
upload_dir = os.path.join(script_dir, 'uploads\\')
allowed_extensions = set(['png', 'jpg', 'jpeg'])

def initWebServices(app: Flask):
    app.config['UPLOAD_DIST'] = upload_dir


def allowed_file(filename) -> str:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

class UploadSummary:
    filename: str
    isUploaded: bool
    msg: str

    def get_uploaded_filename(self) -> str:
        return upload_dir + self.filename

    def __init__(self, filename: str, isUploaded: bool, msg: str):
        self.filename = filename
        self.isUploaded = isUploaded
        self.msg = msg

def upload_file(app: Flask) -> UploadSummary:
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return UploadSummary("", False, "No file part found")
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return UploadSummary("", False, "No files selected")
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_DIST'], filename))
            return UploadSummary(file.filename, True, "")
    
    return UploadSummary("", False, "No files uploaded")