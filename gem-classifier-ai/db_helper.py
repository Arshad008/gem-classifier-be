from flask import Flask, jsonify
from flask_mysqldb import MySQL

def initDb(app: Flask):
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = 'password'
    app.config['MYSQL_DB'] = 'database_name'
    return MySQL(app)


