import azure.functions as func
from azure.functions import AsgiMiddleware

# FastAPI app شما
from knowledge_base.api import app as fastapi_app


# Azure Function App
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


@app.route(
    route="{*path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
)
async def fastapi_proxy(
    req: func.HttpRequest,
    context: func.Context,
) -> func.HttpResponse:
    """
    Proxy all HTTP requests to the FastAPI application.
    """
    return await AsgiMiddleware(fastapi_app).handle_async(req, context)
