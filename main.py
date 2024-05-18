from fastapi import FastAPI, Request, HTTPException
import io
from pydantic import BaseModel
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


class ImageData(BaseModel):
    base64Image: str


@app.post("/upload")
async def upload_image(image_data: ImageData):
    try:
        # Base64 문자열을 디코딩
        image_bytes = base64.b64decode(image_data.base64Image)

        # 이미지 파일을 열어 확인
        image = Image.open(io.BytesIO(image_bytes))

        results = model([image])

        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            # Print box coordinates
            print("Box coordinates:")
            for box in boxes:
                print(box)

        json_results = results_to_json(results)
        # print("Json results:")
        # print(json_results)
        # return json_results

        return {"message": "Image uploaded successfully"}

    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image data") from e


def results_to_json(results):
    return [
        [
            {
                "class": int(box.cls),
                "class_name": model.model.names[int(box.cls)],
                "bbox": [x for x in box.xyxy.tolist()[0]],  # convert bbox results to int from float
                "confidence": float(box.conf),
            }
            for box in result.boxes
        ]
        for result in results
    ]
