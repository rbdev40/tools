release: python manage.py migrate
web: gunicorn dewey.wsgi --timeout 600 --log-file -