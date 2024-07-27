from flask import Flask
from flask_mysqldb import MySQL
from datetime import datetime
from utils import safe_date_cast

import uuid

def initDb(app: Flask):
    app.config['MYSQL_HOST'] = 'localhost'
    app.config['MYSQL_USER'] = 'root'
    app.config['MYSQL_PASSWORD'] = 'Password12345@@@'
    app.config['MYSQL_DB'] = 'gem_classifier_db'
    return MySQL(app)

class JobRecord:
    jobId: str
    note: str
    classifiedClass: str
    imageUrl: str
    createdAt: datetime

    def serialize(self):
        return {
            "jobId" : self.jobId,
            "note" : self.note,
            "classifiedClass" : self.classifiedClass,
            "imageUrl" : self.imageUrl,
            "createdAt" : self.createdAt,
        }

    def __init__(self, jobId: int, note: str, classifiedClass: str, imageUrl: str, createdAt: datetime):
        self.jobId = jobId
        self.note = note
        self.classifiedClass = classifiedClass
        self.imageUrl = imageUrl
        self.createdAt = createdAt


def create_job_record(dbInstance: MySQL, jobId: str, note: str, classifiedClass: str, imageUrl: str, userId: str)-> JobRecord:
    record = JobRecord(jobId, note, classifiedClass, imageUrl, datetime.now())
    cur = dbInstance.connection.cursor()
    cur.execute('''INSERT INTO gem_classification_jobs (`jobId`, `note`, `classifiedClass`, `imageUrl`, `createdAt`, `userId`) VALUES (%s,%s,%s,%s,%s,%s);''',
                (jobId, note, classifiedClass, imageUrl, record.createdAt.isoformat(),userId))
    dbInstance.connection.commit()
    cur.close()
    return record

def get_job_history(dbInstance: MySQL, userId):
    result = []
    cur = dbInstance.connection.cursor()
    cur.execute('''SELECT `jobId`, `note`, `classifiedClass`, `imageUrl`, `createdAt` FROM gem_classification_jobs WHERE (userId = %s)''', [userId])

    for row in cur:
        result.append(JobRecord(row[0], row[1], row[2], row[3], row[4]).serialize())

    return result
class UserRecord:
    userId: str
    name: str
    password: str
    lastLogin: datetime
    createdAt: datetime

    def serialize(self):
        return {
            "userId" : self.userId,
            "name" : self.name,
            "password" : self.password,
            "lastLogin" : self.lastLogin,
            "createdAt" : self.createdAt,
        }

    def fromJson(json):
        return UserRecord(json["userId"] or "", json["name"] or "", json["password"] or "", safe_date_cast(json["lastLogin"]), safe_date_cast(json["createdAt"]))

    def __init__(self, userId: str, name: str, password: str, lastLogin: datetime, createdAt: datetime):
        self.userId = userId
        self.name = name
        self.password = password
        self.lastLogin = lastLogin
        self.createdAt = createdAt

def create_user(dbInstance: MySQL, name: str, password: str)-> UserRecord:
    record = UserRecord(uuid.uuid4(), name, password, datetime.now(), datetime.now())
    cur = dbInstance.connection.cursor()
    cur.execute('''INSERT INTO app_users (`user_id`, `name`, `password`, `last_login`, `created_at`) VALUES (%s,%s,%s,%s,%s);''',
                (record.userId, name, password, record.lastLogin.isoformat(), record.createdAt.isoformat()))
    dbInstance.connection.commit()
    cur.close()
    return record

def get_user_id(dbInstance: MySQL, name: str, password: str)-> UserRecord:
    cur = dbInstance.connection.cursor()
    cur.execute('''SELECT user_id FROM app_users WHERE (name = %s) AND (password = %s)''',
                (name, password))
    result = cur.fetchone()
    cur.close()

    if result:
        return result[0]
    
    return None

def check_for_user_id(dbInstance: MySQL, userId: str)-> bool:
    cur = dbInstance.connection.cursor()
    cur.execute('''SELECT user_id FROM app_users WHERE (user_id = %s)''', [userId])
    result = cur.fetchone()
    cur.close()

    if result is not None:
        return True
    return False