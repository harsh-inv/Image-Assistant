import os

workers = int(os.environ.get('GUNICORN_WORKERS', '2'))
threads = int(os.environ.get('GUNICORN_THREADS', '4'))
timeout = int(os.environ.get('GUNICORN_TIMEOUT', '120'))
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
worker_class = 'sync'
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
accesslog = '-'
errorlog = '-'
loglevel = 'info'
