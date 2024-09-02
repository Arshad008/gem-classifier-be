from flask import Flask
from db_helper import initDb

app = Flask(__name__)
dbInstance = initDb(app)
