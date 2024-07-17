from fastapi import FastAPI, Request, HTTPException
import io
from pydantic import BaseModel
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import uvicorn
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np

app = FastAPI()

global model
global face_detector


@app.on_event("startup")
def on_startup():
    global model
    model = YOLO('detect/yolov8n.pt')
    global face_detector
    face_detector = FaceDetector()


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


def get_face_position_message(bbox, img_width, img_height, confidence, confidence_threshold=0.5):
    if confidence < confidence_threshold:
        return "얼굴 인식 정확도가 낮습니다. 카메라를 조정하십시오."

    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2


    # Define the central region (20% margin)
    margin = 0.2
    left_bound = img_width * margin
    right_bound = img_width * (1 - margin)
    top_bound = img_height * margin
    bottom_bound = img_height * (1 - margin)

    if center_x < left_bound:
        return "카메라를 오른쪽으로 옮기십시오"
    elif center_x > right_bound:
        return "카메라를 왼쪽으로 옮기십시오"
    elif center_y < top_bound:
        return "카메라를 아래로 옮기십시오"
    elif center_y > bottom_bound:
        return "카메라를 위로 옮기십시오"
    else:
        return "얼굴이 중앙에 있습니다"



@app.post("/detect_face")
async def detect_face(image_data: ImageData):
    try:
        # Decode the base64 string to bytes
        image_bytes = base64.b64decode(image_data.base64Image)
        # Convert bytes to numpy array
        image_np = np.frombuffer(image_bytes, np.uint8)
        # Decode the numpy array to an image
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")

        img, bboxes = face_detector.findFaces(img, draw=False)
        img_height, img_width, _ = img.shape

        if bboxes:
            bbox = bboxes[0]["bbox"]
            print(f"bbox: {bbox}")  # 디버그 출력
            confidence = bboxes[0]["score"][0]
            print(f"confidence: {confidence}")  # 디버그 출력
            img_height, img_width, _ = img.shape
            message = get_face_position_message(bbox, img_width, img_height, confidence)
            x, y, w, h = bbox
            center = bboxes[0]["center"]
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            cvzone.putTextRect(img, f'{confidence}%', (x, y - 10))
            cvzone.cornerRect(img, (x, y, w, h))
        else:
            message = "얼굴을 찾을 수 없습니다. 카메라를 조정하십시오."

        window_name = f"Image_{np.random.randint(0, 10000)}"
        cv2.imshow(window_name, img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()


        #return JSONResponse(content={"message": message})
        print("message:",message)
        return {"message": message}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
