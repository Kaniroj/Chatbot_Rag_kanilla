import azure.functions as func
from api import app as fastapi_app  # din FastAPI-app från api.py

app = func.AsgiFunctionApp(app=fastapi_app, http_auth_level=func.AuthLevel.ANONYMOUS)
