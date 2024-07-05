from fastapi import FastAPI, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image
import utils.helping as helping
from pathlib import Path
from typing import Union
import shutil

origins = ["http://localhost:3000"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIRECTORY = "check"
Path(UPLOAD_DIRECTORY).mkdir(parents=True, exist_ok=True)


@app.post("/breed/dog")
async def AutismCheck(image: Union[UploadFile, None] = None):
    try:
       if image is None:
            return {"message": "No image provided"}
       img_path = Path(UPLOAD_DIRECTORY) / image.filename
       with img_path.open("wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
       predicted_breed = helping.predict_breed(img_path)
       return {"message":"Successfully predicted breed", "breed": predicted_breed}
    except Exception as e:
        return {"message":"Error with your code"}



