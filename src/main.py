from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.schemas import RequestModel, ResponseModel
from src.models.pipeline import predict

app = FastAPI(
    title="fit-predict",
    docs_url="/",
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/inference/",
    status_code=status.HTTP_200_OK,
    tags=["Fit-Predict"],
    response_model=ResponseModel
)
async def inference(input_: RequestModel):
    ans = predict(input_)
    return JSONResponse(
        content=ans.model_dump(),
    )


if __name__ == "__main__":
    uvicorn.run(
        app="src.main:app",
        log_level="info",
        workers=1,
        reload=True,
        loop="auto",
        host="localhost",
        port=8000,
    )
