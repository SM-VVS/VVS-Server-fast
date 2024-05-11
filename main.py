from fastapi import FastAPI
from ultralytics import YOLO

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

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
