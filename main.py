from fastapi import FastAPI
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()

global model

@app.on_event("startup")
def on_startup():
    global model
    model = YOLO('detect/yolov8n.pt')

@app.get("/test")
async def detection():
    results = model(['detect/bus.jpg'])
    for result in results:
        print(result)
        #result.show()
        #result.save(filename='detect/output_image.jpg')
    return {"success!"}

@app.post("/photo")
async def photo(photo: str):
    image_data = base64.b64decode(photo)
    image = Image.open(BytesIO(image_data))
    results = model(image)
    for result in results:
        print(result)
    return "success!"

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
