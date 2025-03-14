## 논점 프롬프트 ##
import os 
import sys
from dotenv import load_dotenv
from openai import OpenAI

#지문 받아오기
custom_passage = ""

# .env 파일 로드
load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

# 환경 변수 가져오기
API_KEY = os.getenv("API_KEY")
client = OpenAI(
    api_key=API_KEY,
)

def keypoints_generate_text(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system", "content":system_prompt},
            {"role": "user", "content": user_prompt}
            ]
    )
    return response.choices[0].message.content.strip()

system_prompt = '''
당신은 대한민국 대학수학능력시험의 국어 영역 독서 분야 지문을 생성하는 시험출제 전문가이다. 
학문적 주제를 기반으로 공정하고 객관적인 사실을 다루는 지문을 생성해야 한다. 
하나의 주제를 중심으로 지문 전체의 흐름을 유지하면서도, 동일한 내용이나 유사한 논지를 불필요하게 반복하지 말고 각 문장을 논리적으로 전개해야 한다.
단순한 정보 나열보다 개념 간의 관계를 유기적으로 연결하여 논리적으로 서술해야 한다. 

시험을 치르는 수험생이 지문을 읽고 논리적 추론을 수행할 수 있게 작성해야 한다. 
모든 문장은 한국어로 생성하며 문법적으로 완벽해야 한다.'''

user_prompt = f"""
다음은 한국교육과정평가원 스타일로 생성된 수능 독서 지문입니다.

[생성된 지문]
{custom_passage}

이 지문에서 학생이 반드시 이해해야 할 핵심 논점을 요약하세요.
각 논점은 1~2문장으로 정리하고, 출제 의도를 반영해야 합니다.
출력은 불필요한 문자 없이 한글 문장 현태로만 출력하세요.

"""
key_points = keypoints_generate_text(system_prompt,user_prompt)
print(key_points) 