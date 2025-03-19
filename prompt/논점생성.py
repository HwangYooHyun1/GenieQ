## 논점 프롬프트 ##
import os 
import sys
from dotenv import load_dotenv
from openai import OpenAI

#지문 받아오기
custom_passage = ''''''

# .env 파일 로드
load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

# 환경 변수 가져오기
API_KEY = os.getenv("API_KEY")
client = OpenAI(
    api_key=API_KEY,
)

def keypoints_generate_text(user_prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": user_prompt}
            ]
    )
    return response.choices[0].message.content.strip()


user_prompt = f"""
다음은 한국교육과정평가원 스타일로 생성된 수능 독서 지문입니다.

[생성된 지문]
{custom_passage}

논점이란 해당 글에서 다루는 핵심 주제나 쟁점을 의미한다. 
이는 출제자가 독자에게 전달하고자 하는 주요 메시지나 주장으로, 글의 방향성과 목적을 결정짓는 요소입니다.
이 지문에서 학생이 반드시 이해해야 할 핵심 논점을 요약하라.
각 논점은 1~2문장으로 정리하고, 출제 의도를 반영해야 한다.
출력은 불필요한 문자 없이 한글 문장 형태로만 출력해라.

"""
key_points = keypoints_generate_text(user_prompt)
print(key_points) 