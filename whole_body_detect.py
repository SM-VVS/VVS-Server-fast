import base64
import cv2
from fastapi import HTTPException, APIRouter
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
            message = "얼굴을 찾을 수 없습니다. 카메라를 조정하십시오."
        else:
            sorted(boxes, key=lambda x: x['conf'], reverse=True)
            bbox = result.boxes[0]
            message = f"화면에 {len(boxes)}명이 있습니다."

            # if bbox['conf'] > 0.7:
            #     # 10% margin
            #     margin = 0.1
            #     left_bound = img_width * margin
            #     right_bound = img_width * (1 - margin)
            #     top_bound = img_height * margin
            #     bottom_bound = img_height * (1 - margin)
            #
            #     if bbox['x1'] < left_bound:
            #         message = "카메라를 왼쪽으로 옮기십시오"
            #     elif bbox['x2'] > right_bound:
            #         message = "카메라를 오른쪽으로 옮기십시오"
            #     elif bbox['y1'] < top_bound:
            #         message = "카메라를 위로 옮기십시오"
            #     elif bbox['y2'] > bottom_bound:
            #         message = "카메라를 아래로 옮기십시오"
            #     else:
            #         message = "피사체가 중앙에 있습니다"

                # 이미지 표시
                # cv2.imshow('Detection Results', img)
                # cv2.waitKey(1)
                # cv2.destroyAllWindows()

            # else:
            #     message = "얼굴 인식 정확도가 낮습니다. 카메라를 조정하십시오."

        return {"message": message}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# full height with yolo (multiple)
@router.post("/detect_multiple_whole_body")
async def detect_multiple_whole_body(image_data: ImageData):
    try:
        # Decode the base64 string to bytes
        image_bytes = base64.b64decode(image_data.base64_image)
        # Convert bytes to numpy array
        image_np = np.frombuffer(image_bytes, np.uint8)
        # Decode the numpy array to an image
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")

        # 전면 카메라의 경우 이미지 좌우 반전
        if not image_data.is_back_camera:
            img = cv2.flip(img, 1)

        # apply
        results = model(img)
        message = ""
        ppl = True
        boxes = []

        # 결과 시각화
        for result in results:
            for bbox in result.boxes:
                class_id = int(bbox.cls[0])  # 클래스 ID
                conf = bbox.conf[0]  # 신뢰도
                if class_id == 0:
                    if conf > 0.7:
                        ppl = False
                        x1, y1, x2, y2 = map(int, bbox.xyxy[0])  # 좌표
                        boxes.append({
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "class_id": class_id,
                            "conf": conf
                        })
        if ppl:
            message = "얼굴을 찾을 수 없습니다. 카메라를 조정하십시오."
        else:
            sorted(boxes, key=lambda x: x['x1'])

            message = f"화면에 {len(boxes)}명이 있습니다."
            for box in boxes:
                x1, y1, x2, y2 = box['x1'], box['x2'], box['y1'], box['y2']
                # label = f"{class_id}: {conf:.2f}"  # 레이블 텍스트
                # print(x1, ", ", y1, ", ", x2, ", ", y2)

                # 바운딩 박스
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # 이미지 표시
        # cv2.imshow('Detection Results', img)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()

        # print("message:", message)
        return {"message": message}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))