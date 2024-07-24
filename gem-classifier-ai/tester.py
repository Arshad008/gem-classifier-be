from flask import Flask, jsonify, request
from db_helper import initDb, create_job_record
from flask_mysqldb import MySQL

app = Flask(__name__)
dbInstance = initDb(app)
