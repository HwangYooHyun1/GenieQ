import os
import json
import sys
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
##from prompt.논점생성 import key_points

sys.stdout.reconfigure(encoding='utf-8')
# .env 파일 로드
load_dotenv()

#문항 유형(사실적 읽기, 추론적 읽기, 비판적 읽기, 어휘 및 문법)
type_question = "어휘 및 문법"
#서술 방식(긍정형,부정형)
type_question_detail = "긍정형"
key_points = '''조세는 국가 운영과 공공 서비스의 재정을 마련하는 핵심 수단이며, 효율성과 공평성을 균형 있게 고려하여 설계되어야 한다. 효율성을 위해 세부담 조절과 재정 안정성을 보장해야 하며, 공평성을 위해 소득 수준에 따른 정당한 조세 부담 분배와 탈세 방지 대책이 중요하다. 다양한 이해관계자와 전문가의 의견을 수렴하고, 다양한 학문 분야의 분석을 결합하여 조세 제도의 정당성을 검토해야 한다. 국제 무역의 확대와 관련하여 다국적 기업의 조세 회피와 국가 간 조세 경쟁 문제를 해결하기 위한 법적·제도적 협력이 필요하다.'''
#지문
custom_passage = '''사회에서 조세는 국가 운영과 공공 서비스의 재정을 마련하는 핵심 수단으로, 효율적인 자원 분배와 공평한 부담을 동시에 추구해야 하는 과제를 안고 있다. 이러한 조세 체계는 국민 소득과 재산의 일정 비율을 세금으로 거두어 사회적 필요를 충족하는 예산을 구성하도록 돕는다. 특히 공공 기반 시설이나 교육·보건 서비스 같은 영역에서는 적절한 재원 확보가 필수적이므로, 조세 제도 설계 시 효율성과 공평성을 균형 있게 고려할 필요가 대두된다. 이를 위해 입법 기관과 정부 부처는 세율 결정, 과세 기준 설정, 세액 공제 제도 운용 등 다양한 측면에서 정교한 정책적 판단을 내린다.

효율성 관점에서 보면, 조세는 시장에서 발생하는 외부효과를 교정하거나 정부의 공공재 공급을 위한 재원을 확보하는 기능을 수행한다. 만약 세금이 과도하게 부과되거나 누진 체계를 지나치게 강화하면, 높은 세부담으로 인해 개인과 기업이 생산과 투자 활동을 위축시킬 가능성이 제기된다. 반대로 세율이 지나치게 낮으면, 공공 서비스 제공을 위한 예산이 부족해져 전반적인 사회적 편익이 저해될 위험이 커진다. 따라서 효율성 증대를 위해서는 세부담 여력을 감안하면서도 재정 안정성을 보장하는 조절 기제가 마련되어야 한다.

공평성 관점에서는 조세 부담이 소득 수준이나 자산 규모에 따라 정당하게 분배되는지를 점검하는 과정이 중요하다. 일반적으로 조세 공평성은 수직적 형평성과 수평적 형평성으로 나뉘며, 전자는 고소득자가 더 많은 세금을 내는 구조, 후자는 동일 소득자 간 부담의 균등성을 의미한다. 그러나 지나친 누진 과세는 경제 주체의 창의적 활동을 저해할 가능성이 있으므로, 적정 수준의 공평성을 확보하면서도 생산 동기를 꺾지 않는 절충안이 필요하다. 이를 실현하기 위해 조세 감면 정책이나 세 크레디트 제도를 활용하여 소득 하위 계층의 부담을 완화하는 동시에, 탈세 방지 대책을 철저히 시행하는 방안이 검토된다.

한편, 조세 관련 제도나 정책을 수립할 때는 다양한 이해관계자와 전문가의 의견을 수렴하고, 구체적인 통계 자료를 토대로 합리적 기준을 설정하는 과정이 필수적이다. 이를 통해 예산 확보와 분배가 효율적으로 진행되면, 공공 부문에서 추진하는 여러 영역의 사업성이 제고되고 궁극적으로 사회 전체의 성장 동력이 강화될 수 있다. 아울러 국제 무역 규모가 확대되는 상황에서는 다국적 기업의 조세 회피 문제나 국가 간 조세 경쟁이 발생하므로, 법적·제도적 협력을 심화하는 방향도 모색되어야 한다. 조세의 효율성과 공평성을 동시에 추구하는 정책 설계가 이처럼 복합적인 측면을 지니고 있다는 점에서, 넓은 범위의 검토와 지속적인 제도 개선이 요구된다.

한편, 조세 제도를 평가할 때는 법학, 경제학, 사회학 등 다양한 학문 분야의 분석이 결합되어야 하며, 제도적 정당성도 면밀히 검토될 필요가 있다. 이를 통해 사회 구성원들이 납세의 의의를 충분히 인식하고, 부과된 세금이 공익적 목적을 위해 합리적으로 사용된다는 신뢰를 형성할 수 있다. 특히 소득 재분배 효과가 큰 조세 제도의 경우, 공공 정책 전반에 대한 국민 지지도를 높이는 역할도 수행한다. 이처럼 여러 요인을 균형 있게 고려하는 접근 방식이 뒷받침될 때, 조세의 효율성과 공평성 간 조화를 현실화할 수 있다는 점이 주목된다.'''
#문항 유형
question_example = "문맥상 'OOO'과 바꾸어 쓰기에 적절한 것은?"

# API 키 설정
API_KEY = os.getenv("API_KEY")
os.environ['OPENAI_API_KEY'] = API_KEY

client = OpenAI(api_key=API_KEY)

def generate_text(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model="o1",
        messages=[
            {"role": "system", "content": system_prompt},
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
        generated_question = response_json.get("generated_question", "질문 생성 실패")
        generated_option = response_json.get("generated_option", ["선지 없음"] * 5)
        generated_answer = response_json.get("generated_answer", "정답 없음")
        generated_description = response_json.get("generated_description", "해설 없음")
        
        return generated_question, generated_option, generated_answer, generated_description
    
    except json.JSONDecodeError:
        # JSON 변환 실패 시 오류 로그 출력 및 원본 응답 표시
        print("JSON 파싱 오류 발생! 원본 응답을 확인하세요:")
        print(response_text)
        return response_text, ["오류 발생"], "오류 발생", "JSON 파싱 오류 - 원본 응답 확인 필요"
    

q_system_text = '''
    당신은 대한민국 대학수학능력시험의 국어 영역 독서 분야 문항을 생성하는 시험출제 전문가이다. 
    문항은 지문에서 측정하고자 하는 내용을 정확히 반영하고, 핵심 내용을 간결하고 구조적이며 체계적으로 구성해야 한다. 

    선택지를 작성할 때는 문법적, 논리적으로 지문과 일치하도록 하며, 정답과 오답이 명확하게 구별되도록 해야한다.
    정답의 위치는 무작위로 배치하여 특정 패턴이 드러나지 않도록 하고, 다른 문항과 중복되지 않도록 해야 한다. 
    또한 단순히 특정 어휘를 대체하는 방식으로 오답을 구성하면 안된다. 
''' 

# OpenAI 임베딩 모델 초기화
embeddings_model = OpenAIEmbeddings()

# FAISS 인덱스 불러오기
faiss_question_all_path = "./faiss_index/faiss_index_question_all"
vector_db_question_all = FAISS.load_local(faiss_question_all_path, embeddings_model, allow_dangerous_deserialization=True)

# 유사한 문서 검색
question_guidelines = vector_db_question_all.similarity_search(type_question, k=1)

q_user_prompt = f"""

다음은 {type_question} 유형의 문항과 선지를 작성할 때 반드시 고려해야 할 기준이다.

{question_guidelines}

다음으로 제시하는 지문과 문항 유형을 바탕으로 5개의 선지로 이루어진 문항 1개를 작성해라.
지문의 논점을 반영하여 문항을 생성해라.
[지문]
{custom_passage}

[논점]
{key_points}

[문항 예시]
{question_example}

문항을 출력하는 형식은 아래와 같다. 
정답 및 해설은 선지 번호( ①, ②, ③, ④, ⑤)를 활용하여 정답 선지의 근거와 오답의 틀린 이유를 포함한 상세 해설을 최소 100자, 최대 200자로 출력해야 한다. 
선지 출력 결과 안에는 선지 번호( ①, ②, ③, ④, ⑤)를 포함하지 말아라.
문항은 불필요한 텍스트 없이 아래의 json 형식으로만 출력해라. 
정답은 선지 번호( ①, ②, ③, ④, ⑤)로 출력해라.

```json
{{
    "generated_question": "질문",
    "generated_option": ["선지1", "선지2", "선지3", "선지4", "선지5"],
    "generated_answer": "정답",
    "generated_description": "해설"
}}
```

"""

question, options, answer, description = generate_text(q_system_text, q_user_prompt)
print(question)
print(options)
print(answer)
print(description)
