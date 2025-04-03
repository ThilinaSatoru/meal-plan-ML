# Run

## `run.py `

- pip install flask flask-sqlalchemy flask-migrate python-dotenv
- $env:FLASK_APP="run.py"
- flask db init
- flask db migrate -m "Initial migration"
- flask db upgrade

## http://127.0.0.1:5000/docs