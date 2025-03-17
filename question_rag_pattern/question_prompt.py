import os
import json
import sys
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


sys.stdout.reconfigure(encoding='utf-8')
# .env 파일 로드
load_dotenv()

#문항 유형(사실적 읽기, 추론적 읽기, 비판적 읽기, 어휘 및 문법)
type_question = "어휘 및 문법"
#서술 방식(긍정형,부정형)
type_question_detail = "부정형"
#지문
custom_passage = "투명한 유리창은 외부의 빛을 거의 그대로 통과시키기 때문에 강렬한 여름 햇살이 유리창을 통과해서 실내로 들어오는 경우, 실내의 온도가 점점 올라간다. 이를 방지하기 위해 일반적으로 이용되는 방법이 커튼을 이용해서 실내로 들어오는 빛을 차단하는 것이다. 그런데 커튼에 의해 흡수된 빛은 커튼의 온도를 올리고 이는 다시 방의 온도를 상승시키므로, 방 전체의 열 출입이라는 관점에서 보면 빛 차단의 효과는 제한적이다. 이를 보완하기 위해서 커튼 역할을 하는 유리창의 필요성을 생각해 볼 수 있다. 그러나 빛을 계속해서 차단하는 유리창은 유리창 본래의 역할을 수행할 수 없기 때문에 투명 상태와 불투명 상태 사이에서 자유롭게 변환이 가능한 조광 유리가 고안되었다. \n조광 유리를 만드는 방법 중 가장 많이 사용되는 것은 전기적 변환 방식인데, 2장의 투명 전극 사이를 용액으로 채우고 전압을 가해 용액의 색을 변환시키는 용액형 방식이 대표적이다. 이 방식의 기본 원리는 전자를 받았을 때와 전자를 내주었을 때 광학적 성질이 달라져서 색이 변화하는 물질을 이용하는 것이다. 투명 전극 사이에 채워진 용액에는 각각 환원 착색제와 산화 착색제 역할을 하는 두 가지 종류의 물질이 용해되어 있다. 전압이 가해지지 않은 상태에서 환원 착색제와 산화 착색제는 모두 투명한 상태로 유지된다. 하지만 2장의 투명 전극에 전압을 가하면 음극에서 양극으로 전자가 이동하면서, 여분의 전자를 받은 환원 착색제와 전자를 빼앗긴 산화 착색제가 초록색과 빨간색 계열 빛의 대부분을 흡수한다. 이때 푸른색 빛의 일부만이 유리를 통과하게 되면서 유리는 푸른색을 띤 불투명한 상태가 된다. \n전기적 변환 방식으로 제작한 조광 유리를 실제 건물에 설치하면, 일반 투명 유리를 설치한 건물에 비해 냉방에 드는 에너지를 30% 이상 절감할 수 있다. 또한 조광 유리에서는 전압의 세기와 투명도가 반 비례하기 때문에 전압의 세기를 달리하여 원하는 만큼 투명도를 조절할 수 있다. 그러나 복잡한 구조로 인해 제작 및 유지 비용이 많이 들기 때문에 에너지 절약으로 인한 비용 절감 효과가 상쇄되어 비경제적이라는 점, 유리가 클수록 투명 상태와 불투명 상태 간의 전환에 걸리는 시간이 길다는 점이 한계로 지적된다. 그리고 이 방식에 의한 조광 유리는 외부에서 유입되는 대부분의 빛을 흡수하기 때문에 불투명화에 사용되는 용액의 온도가 상승하게 되고, 그것이 실내 열로 다시 방사되는 단점도 있다. \n그렇다면 유리창이 거울의 기능을 갖도록 하여 빛을 반사시킴으로써 실내로 유입되는 열을 크게 줄일 수 있지 않을까? 이러한 과제와 관련하여 가장 활발히 연구되는 방식이 수소 가스를 주로 이용하는 가스 변환 방식이다. 기존의 가스 변환 방식은 2장의 유리 사이의 공간에 수소 가스를 충전하여 유리의 상태를 변환시키는 방식으로, 유리 2장의 두께로 인해 활용에 제약이 있었다. 하지만 새로운 가스 변환 방식에서는 1장의 유리 표면에 마그네슘과 이트륨이 혼합된 두께 약 40nm 정도의 박막을 입히고, 여기에 다시 수소의 흡수와 탈착을 촉진하는 촉매의 역할을 하며 수분으로 인해 발생할 수 있는 박막의 산화를 막는 팔라듐 박막을 입힌 다음, 얇은 투명 시트를 덧씌운다. 마그네슘과 이트륨의 혼합 박막은 불투명한 금속 상태에서 거울의 기능을 하여 빛을 반사하는 역할을 한다. 유리 표면의 박막과 얇은 투명 시트 사이에는 평균 0.1mm 정도의 간격이 있는데, 이 틈새에 공기 중의 수분을 전기 분해함으로써 발생된 수소 가스가 차면 혼합 박막이 투명해진다. 그리고 수소 가스 공급이 중단되면 혼합 박막과 투명 시트 사이의 수소가 대기 중의 산소와 반응하여 수증기가 되어 빠져나가기 때문에 혼합 박막은 불투명한 상태로 돌아간다. 이 조광 유리는 공기 중의 수분을 이용하여 발생시킨 소량의 수소만으로도 상태의 전환이 가능하기 때문에 제작 및 유지 비용이 적어 전기적 변환 방식에 비해 경제적이며, 자동차 유리와 같은 실용적 분야에서 다양하게 활용될 수 있을 것으로 기대되고 있다."
#문항 유형
question_example = "'OOO'의 관점에서 'OOO'을 이해한 내용으로 적절하지 않은 것은?"
#논점(논점 프롬프트 활용)
key_points = '''1. 조광 유리는 투명한 유리창의 단점을 보완하여 실내 온도 조절에 도움을 주며, 전기적 변환 방식과 가스 변환 방식을 통해 구현된다.
2. 전기적 변환 방식은 전압 변화를 통해 유리의 투명도를 조절하지만, 높은 제작 비용과 실내 열 방사 문제를 가진다.
3. 가스 변환 방식은 수소 가스를 이용하여 에너지 절감 효과를 제공하며, 경제적이고 다양한 실용적 분야에 활용될 수 있다.'''

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
faiss_question_all_path = "./question_rag_pattern/faiss_index/faiss_index_question_all"
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
문항은 불필요한 텍스트 없이 아래의 json 형식으로만 출력해라.

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
