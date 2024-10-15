from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://ronald1:123@localhost/fire_prediction'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class FireIncident(db.Model):
    __tablename__ = 'fire_incidents'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    location = db.Column(db.String(255), nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    wind_speed = db.Column(db.Float, nullable=False)
    other_factors = db.Column(db.JSON, nullable=True)
    fire_occurred = db.Column(db.Boolean, nullable=False)
