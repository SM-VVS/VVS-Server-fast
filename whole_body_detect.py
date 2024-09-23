import base64
import cv2
from fastapi import HTTPException, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from ultralytics import YOLO

router = APIRouter()

# get YOLOv8 model
yolo_weights_path = 'yolov8n.pt'
model = YOLO(yolo_weights_path)

class ImageData(BaseModel):
    base64_image: str
    is_back_camera: bool

# full height with yolo (single)
@router.post("/detect_whole_body")
async def detect_whole_body(image_data: ImageData):
    try:
        # Decode the base64 string to bytes
        image_bytes = base64.b64decode(image_data.base64_image)
        # Convert bytes to numpy array
        image_np = np.frombuffer(image_bytes, np.uint8)
        # Decode the numpy array to an image
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        img_height, img_width, _ = img.shape

        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")

        # 전면 카메라의 경우 이미지 좌우 반전
        if not image_data.is_back_camera:
            img = cv2.flip(img, 1)

        # apply
        results = model.predict(source=img, show=False, classes=0)
        message = ""
        boxes = []

        # 결과 시각화
        for result in results:
            for bbox in result.boxes:
                class_id = int(bbox.cls[0])  # 클래스 ID
                conf = bbox.conf[0]  # 신뢰도
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])  # 좌표
                boxes.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class_id": class_id,
                    "conf": conf
                })

        plen = len(boxes)
        if plen == 0:
            message = "전신을 찾을 수 없습니다. 카메라를 조정하십시오."
        else:
            sorted(boxes, key=lambda x: x['conf'], reverse=True)
            bbox = boxes[0]
            # message = f"화면에 {len(boxes)}명이 있습니다."

            if bbox['conf'] > 0.7:
                # 5% margin
                margin = 0.05
                left_bound = img_width * margin
                right_bound = img_width * (1 - margin)
                top_bound = img_height * margin
                bottom_bound = img_height * (1 - margin)

                if bbox['x1'] < left_bound:
                    message = "카메라를 왼쪽으로 옮기십시오"
                elif bbox['x2'] > right_bound:
                    message = "카메라를 오른쪽으로 옮기십시오"
                elif bbox['y1'] < top_bound:
                    message = "카메라를 위로 옮기십시오"
                elif bbox['y2'] > bottom_bound:
                    message = "카메라를 아래로 옮기십시오"
                else:
                    message = "촬영하세요"

                # 이미지 표시
                # cv2.imshow('Detection Results', img)
                # cv2.waitKey(1)
                # cv2.destroyAllWindows()

            else:
                message = "얼굴 인식 정확도가 낮습니다. 카메라를 조정하십시오."

        return {"message": message}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class MultiFaceRequest(BaseModel):
    base64_image: str
    required_faces: int
    is_back_camera: bool

# full height with yolo (multiple)
@router.post("/detect_multiple_whole_body")
async def detect_multiple_whole_body(request: MultiFaceRequest):
    try:
        image_bytes = base64.b64decode(request.base64_image)
        image_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        img_height, img_width, _ = img.shape

        if img is None:
            raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다.")

        # 전면 카메라의 경우 이미지 좌우 반전
        if not request.is_back_camera:
            img = cv2.flip(img, 1)

        # apply
        results = model(img)
        message = ""
        boxes = []

        # 결과 시각화
        for result in results:
            for bbox in result.boxes:
                class_id = int(bbox.cls[0])  # 클래스 ID
                conf = bbox.conf[0]  # 신뢰도
                if conf > 0.7:
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0])  # 좌표
                    boxes.append({
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "class_id": class_id,
                        "conf": conf
                    })

        plen = len(boxes)
        if plen == 0:
            message = "얼굴을 찾을 수 없습니다. 카메라를 조정하십시오."
        elif len(boxes) != request.required_faces:
            message = f"화면에 {len(boxes)}명이 있습니다."
        else:
            sorted(boxes, key=lambda x: x['x1'], reverse=True)

            # 5% margin
            margin = 0.05
            left_bound = img_width * margin
            right_bound = img_width * (1 - margin)
            top_bound = img_height * margin
            bottom_bound = img_height * (1 - margin)

            for idx, bbox in enumerate(boxes):
                direction_message = []

                if bbox['x1'] < left_bound:
                    direction_message.append("오른쪽으로")
                elif bbox['x2'] > right_bound:
                    direction_message.append("왼쪽으로")

                if bbox['y1'] < top_bound:
                    direction_message.append("아래로")
                elif bbox['y2'] > bottom_bound:
                    direction_message.append("위로")

                if direction_message:
                    message.append(f"왼쪽에서 {idx+1}번째 분 {' 그리고 '.join(direction_message)}")

            if message:
                message.append("이동해 주세요")
            else:
                message = "촬영하세요"

        return {"message": message}

    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=400)
