import base64
import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
import openai
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

router = APIRouter()


class ImageDescribeRequest(BaseModel):
    base64_image: str
    target_object: str


@router.post("/describe_image")
async def describe_image(image: ImageDescribeRequest):
    base64_image = image.base64_image
    target_object = image.target_object

    try:
        api_key = os.getenv('OPENAI_API_KEY')
        client = OpenAI(
            api_key=api_key
        )

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": f"이 사진을 {target_object}에 집중해서 한글로 설명해 줘"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            model="gpt-4o",
            max_tokens=200
        )
        answer = response.choices[0].message.content
        return {"message": answer}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)


@router.post("/test")
def post_gpt():
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        client = OpenAI(
            api_key=api_key
        )

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Say this is a test",
                }
            ],
            model="gpt-4o",
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


@router.post("/test2")
async def describe_image():
    image_path = "detect/photo.jpeg"
    base64_image = encode_image(image_path)
    target_object = "people"

    try:
        api_key = os.getenv('OPENAI_API_KEY')
        client = OpenAI(
            api_key=api_key
        )

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": f"이 사진을 {target_object}에 집중해서 한글로 설명해줘"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            model="gpt-4o",
            max_tokens=300
        )
        answer = response.choices[0].message.content
        return {"message": answer}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)

