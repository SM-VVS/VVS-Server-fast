import base64
import io

import cv2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np

router = APIRouter()


# get YOLOv8 model
yolo_weights_path = 'yolov8n.pt'
yolo = YOLO(yolo_weights_path)

class ObjectDetectRequest(BaseModel):
    base64_image: str
    is_back_camera: bool
    target_object: int


@router.post("/detect_object")
async def detect_object(image_data: ObjectDetectRequest):
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

        results = yolo.predict(source=img, classes=image_data.target_object)

        message = ""
        img_width, img_height, _ = img.shape

        if results and any(result.boxes for result in results):
            for result in results:
                for bbox in result.boxes:
                    # 바운딩 박스 좌표 추출 (x1, y1: 좌상단, x2, y2: 우하단)
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0])

                    # 바운딩 박스가 화면을 벗어난 경우
                    if x1 <= 0:
                        message = "카메라를 왼쪽으로 옮기십시오"
                    elif x2 >= img_width:
                        message = "카메라를 오른쪽으로 옮기십시오"
                    elif y1 <= 0:
                        message = "카메라를 위로 옮기십시오"
                    elif y2 >= img_height:
                        message = "카메라를 아래로 옮기십시오"
                    else:
                        message = "촬영하세요."

                    # 바운딩 박스 그리기 (디버깅 또는 시각화를 위해)
                    #cv2.rectangle(np.array(img), (x1, y1), (x2, y2), (0, 255, 0), 2)
                result_img = result.plot()  # 결과 이미지를 반환 (YOLOv8 기준)
                cv2.imshow("Detected Object", result_img)
                cv2.waitKey(1000)  # 1초 대기
                cv2.destroyAllWindows()  # 창 닫기
        else:
            message = "물체를 인식 할 수 없습니다. 카메라를 조정하십시오."


        # 프론트엔드로 메시지 전송
        return {"message": message}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
