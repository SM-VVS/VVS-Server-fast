from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import base64
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np

app = FastAPI()

global face_detector

def on_startup_face_detect():
    global face_detector
    face_detector = FaceDetector()


class ImageData(BaseModel):
    base64_image: str
    is_back_camera: bool


@app.post("/detect_face")
async def detect_face(image_data: ImageData):
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

        img, bboxes = face_detector.findFaces(img, draw=True)
        img_height, img_width, _ = img.shape

        if bboxes:
            bbox = bboxes[0]["bbox"]
            confidence = bboxes[0]["score"][0]
            #x, y는 바운딩 박스 왼쪽 상단 좌표
            x, y, w, h = bbox
            right_x = x + w
            bottom_y = y + h

            # 중앙 범위 정의 (20% margin)
            margin = 0.2
            left_bound = img_width * margin
            right_bound = img_width * (1 - margin)
            top_bound = img_height * margin
            bottom_bound = img_height * (1 - margin)

            # 얼굴이 너무 클 때 기준 정의 (얼굴이 이미지의 50% 이상을 차지할 때)
            large_face_threshold = 0.5

            #음성 안내 메세지
            if confidence >= 0.5:
                if w / img_width > large_face_threshold or h / img_height > large_face_threshold:
                    message = "얼굴이 너무 가깝습니다. 카메라를 뒤로 옮기십시오."
                elif x < left_bound:
                    message = "카메라를 왼쪽으로 옮기십시오"
                elif right_x > right_bound:
                    message = "카메라를 오른쪽으로 옮기십시오"
                elif y < top_bound:
                    message = "카메라를 위로 옮기십시오"
                elif bottom_y > bottom_bound:
                    message = "카메라를 아래로 옮기십시오"
                else:
                    message = "얼굴이 중앙에 있습니다"
            else:
                message = "얼굴 인식 정확도가 낮습니다. 카메라를 조정하십시오."

            # 중앙 범위 그리기
            cv2.line(img, (int(left_bound), 0), (int(left_bound), img_height), (0, 255, 0), 2)
            cv2.line(img, (int(right_bound), 0), (int(right_bound), img_height), (0, 255, 0), 2)
            cv2.line(img, (0, int(top_bound)), (img_width, int(top_bound)), (0, 255, 0), 2)
            cv2.line(img, (0, int(bottom_bound)), (img_width, int(bottom_bound)), (0, 255, 0), 2)

        else:
            message = "얼굴을 찾을 수 없습니다. 카메라를 조정하십시오."

        #인식 후 결과 보기
        window_name = f"Image_{np.random.randint(0, 10000)}"
        print(window_name)
        cv2.imshow(window_name, img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        print("message:", message)
        return {"message": message}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class MultiFaceRequest(BaseModel):
    base64_image: str
    required_faces: int


@app.post("/detect_multiple_face")
async def detect_multiple_face(request: MultiFaceRequest):
    try:
        # 디코딩
        image_bytes = base64.b64decode(request.base64_image)
        image_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="이미지를 디코딩할 수 없습니다.")

        img, bboxes = face_detector.findFaces(img, draw=True)

        # 정확도가 70% 이상인 얼굴만 카운트
        face_count = sum(1 for bbox in bboxes if bbox['score'][0] >= 0.7)

        required_faces = request.required_faces

        # 인식 후 결과 보기
        window_name = f"Image_{np.random.randint(0, 10000)}"
        print(window_name)
        cv2.imshow(window_name, img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        if face_count < required_faces:
            message = f"화면에 {face_count}명이 있습니다. 안쪽으로 더 들어와 주십시오"
            return {"message": message}
        else:
            message = f"화면에 {face_count}명이 있습니다."
            return {"message": message}

    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=400)