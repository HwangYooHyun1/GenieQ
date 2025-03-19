## 주제 및 키워드 프롬프트##
import os 
import sys
import json
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

def keypoints_generate_text(user_prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": user_prompt}
            ]
    )
    response_text = response.choices[0].message.content.strip()
    # 코드 블록 구분자 제거: 만약 응답이 ```json 으로 시작하면 제거
    if response_text.startswith("```json"):
        # 첫 줄(코드블록 시작) 제거
        response_text = response_text.split("\n", 1)[1]
    # 마지막 코드 블록 닫는 구분자 제거
    if response_text.endswith("```"):
        response_text = response_text.rsplit("```", 1)[0]
    
    try:
        # JSON 변환 시도
        response_json = json.loads(response_text)
        
        # JSON 응답에서 필요한 값 추출
        type_passage = response_json.get("type_passage", "논점 생성 실패")
        keyword = response_json.get("keyword", ["키워드 없음"] * 3)
        
        return type_passage, keyword
    
    except json.JSONDecodeError:
        # JSON 변환 실패 시 오류 로그 출력 및 원본 응답 표시
        print("JSON 파싱 오류 발생! 원본 응답을 확인하세요:")
        print(response_text)
        return response_text, ["오류 발생"], "오류 발생", "JSON 파싱 오류 - 원본 응답 확인 필요"


user_prompt = f"""
아래의 지문을 5가지 주제 (인문, 예술, 사회, 기술, 과학 중 선택) 중 하나로 분류하세요.
그리고 15자 이내로 요약한 핵심 키워드를 1~3개 뽑아주세요. 쉼표(,)로 구분해주세요.

    [주제 분류 기준] 
    인문: 인간의 존재와 관련된 문제, 그리고 인간의 사상과 문화 등을 다루고 있는 글이다. 인간의 본질이나 정신세계, 그리고 인간의 행위에 대한 이해를 목적으로 하는 글이다. 인간과 세계의 본질과 관련된 글, 인간의 행위 규범과 관련된 글, 인간의 의식 세계와 관련된 글, 사유의 형식이나 법칙과 관련된 글, 그리고 역사나 종교와 관련된 글 등을 포함한다.
    예술: 미의 본질이나 미를 추구하는 인간의 다양한 예술 행위에 대해 다루고 있는 글이다. 예술의 본질과 다양한 예술 행위의 특징을 이해하는 한편, 예술 작품을 수용하는 미적 안목을 향상하는 데 도움을 주기 위한 글이다. 예술의 본질에 대해 논의하는 글, 다양한 예술 행위의 특징을 설명하는 글, 주요 예술가나 예술 작품을 비평하는 글, 예술 사조에 대해 설명하는 글 등을 포함한다.
    사회: 사회에서 일어나거나 일어날 수 있는 다양한 문제를 소개하고 해결하는 방안을 제시하 는 글이다. 사회 현상이나 문화 현상을 다양한 관점에서 논리적, 체계적으로 설명하는 글이다. 법을 다룬 법학, 사회 제도 및 사회의 다양한 현상을 연구하는 사회학, 기업의 경영을 다룬 경영학, 경제 문제 및 경제 활동을 설명하는 경제학, 생물로서의 인간을 종합적으로 연구하는 인류학, 사회 구성원에 의해 이루어진 생활 양식 및 그와 관련하여 일어나는 여러 현상들을 연구하는 문화학과 관련된 글 등을 포함한다.
    기술: 인간의 삶을 편리하게 하는 산업 기술, 생활 기술 등 다양한 분야의 기술을 설명하는 글이다. 특정 과학 이론을 바탕으로 장치나 시스템에 적용되는 원리와 작동 과정, 한계 등을 구체적으로 서술한 글이다. 전기와 전자의 원리를 이용한 공학 기술, 컴퓨터를 이용한 공학 기술, 화학이나 생명 과학과 결합된 공학 기술, 토목이나 건축에 활용되는 토목건축 공학 기술과 관련된 글 둥을 포함한다.
    과학: 자연 과학적 시각으로 물질계와 생태계, 우주를 탐구하는 인간의 정신 활동을 담고 있는 글이다. 수에 관하여 연구하는 수학, 물질의 물리적 성질과 운동 형태 등을 연구하는 물리학, 물질의 조성과 구조 · 성질 등을 연구하는 화학, 생물의 구조와 기능을 과학적으로 연구하는 생명 과학, 지구 및 천체를 연구하는 지구 과학과 관련된 글 등을 포함한다.

    [지문]
    {custom_passage}
    
출력은 불필요한 텍스트 없이 아래의 json 형식으로만 출력하세요.
```json
{{
  "type_passage": "string",
  "keyword": ["string", "string", "string"]
}}
```
"""
type_passage, keyword = keypoints_generate_text(user_prompt)
print(type_passage)
print(keyword)