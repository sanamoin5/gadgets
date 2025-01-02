import os
from decouple import config


class VarsConfig:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATABASE_URL = config('DATABASE_URL', default='')
    SECRET_KEY = config('SECRET_KEY', default='')
    POSTGRES_USER = config('POSTGRES_USER', default='')
    POSTGRES_HOST = config('POSTGRES_HOST', default='')
    POSTGRES_PORT = config('POSTGRES_PORT', default='')
    POSTGRES_PASSWORD = config('POSTGRES_PASSWORD', default='')
    POSTGRES_DB_NAME = config('POSTGRES_DB', default='')

