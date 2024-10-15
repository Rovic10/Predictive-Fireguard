from models import db, FireIncident, app
from datetime import date

with app.app_context():
    db.create_all()

    # Sample data
    fire_incident1 = FireIncident(
        date=date(2024, 7, 21),
        location="Location A",
        temperature=35.5,
        humidity=60.0,
        wind_speed=15.0,
        other_factors={"air_quality": "good"},
        fire_occurred=True
    )
    fire_incident2 = FireIncident(
        date=date(2024, 7, 22),
        location="Location B",
        temperature=30.2,
        humidity=70.0,
        wind_speed=10.5,
        other_factors={"air_quality": "moderate"},
        fire_occurred=False
    )
    fire_incident3 = FireIncident(
        date=date(2024, 7, 23),
        location="Location C",
        temperature=32.1,
        humidity=65.0,
        wind_speed=12.0,
        other_factors={"air_quality": "poor"},
        fire_occurred=True
    )

    db.session.add(fire_incident1)
    db.session.add(fire_incident2)
    db.session.add(fire_incident3)
    db.session.commit()
