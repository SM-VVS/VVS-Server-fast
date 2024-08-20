from fastapi import FastAPI, Request, HTTPException
import io
from pydantic import BaseModel
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import uvicorn
import cv2
from face_detect import router as face_detect_router, on_startup_face_detect
from whole_body_detect import router as whole_body_detect_router
from image_describe import router as image_describe_router

app = FastAPI()

global model


@app.on_event("startup")
def on_startup():
    global model
    model = YOLO('detect/yolov8n.pt')
    on_startup_face_detect()


# face_detect.py, whole_body_detect.py의 라우터를 포함
app.include_router(face_detect_router, prefix="/face_detect")
app.include_router(whole_body_detect_router, prefix="/whole_body_detect")
app.include_router(image_describe_router,prefix="/image_describe")

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
            #객체 정보
            json_result = boxes_to_json(boxes)
            print("Json results:")
            print(json_result)
            result.show()
            # Display result with boxes
            result.plot()
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return json_result

        # json_results = results_to_json(results)
        # print("Json results:")
        # print(json_results)
        # return json_results

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


def boxes_to_json(boxes):
    return [
        {
            "class": int(box.cls),
            "class_name": model.model.names[int(box.cls)],
            "bbox": [x for x in box.xyxy.tolist()[0]],  # convert bbox results to int from float
            "confidence": float(box.conf),
        }
        for box in boxes
    ]


@app.get("/test2")
async def detection():
    results = model(['detect/bus.jpg'])
    json_results = results_to_json(results)

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Class probabilities for classification outputs
        # Print box coordinates
        print("Box coordinates:")

        for box in boxes:
            print(box)

    return json_results




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
