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


# full height with yolo
@router.post("/detect_whole_body")
async def detect_whole_body(image_data: ImageData):
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

        # 결과 시각화
        for result in results:
            for bbox in result.boxes:
                class_id = int(bbox.cls[0])  # 클래스 ID
                conf = bbox.conf[0]  # 신뢰도
                if class_id == 0:
                    if conf > 0.5:
                        x1, y1, x2, y2 = map(int, bbox.xyxy[0])  # 좌표
                        label = f"{class_id}: {conf:.2f}"  # 레이블 텍스트

                        print(x1, ", ", y1, ", ", x2, ", ", y2)

                        # 바운딩 박스
                        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if x1 == 0:
                            message = "카메라를 왼쪽으로 옮기십시오"
                        elif y1 == 0:
                            message = "카메라를 아래로 옮기십시오"
                        elif x2 == 470:
                            message = "카메라를 오른쪽으로 옮기십시오"
                        elif y2 == 630:
                            message = "카메라를 위로 옮기십시오"

                    else:
                        message = "얼굴 인식 정확도가 낮습니다. 카메라를 조정하십시오."


        # 이미지 표시
        # cv2.imshow('Detection Results', img)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        print("message:", message)
        return {"message": message}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))