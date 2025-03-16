## 지문 프롬프트 ##
## 미완성 수정중##
import os 
import sys
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# .env 파일 로드
load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

#지문 주제(분야)
type_passage = ""
##지문 제재 
keyword = ""

# 환경 변수 가져오기
API_KEY = os.getenv("API_KEY")
client = OpenAI(
    api_key=API_KEY,
)

def passage_generate_text(system_prompt, user_prompt):
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

# OpenAI 임베딩 모델 초기화
embeddings_model = OpenAIEmbeddings()

# FAISS 인덱스 불러오기
faiss_passage_path = "./passage_faiss/faiss_index/faiss_index_passage"
vector_db_passage = FAISS.load_local(faiss_passage_path, embeddings_model, allow_dangerous_deserialization=True)

# 유사한 문서 검색
passage_guidelines = vector_db_passage.similarity_search(keyword, k=1)

user_prompt = f'''  
다음은 {type_passage} 분야의 출제 경향 및 작성 원칙이다.
{passage_guidelines}

다음은 문장 구성 및 지문 작성 원칙에 대한 정리이다.
모든 문장은 문어체로 논리적이고 객관적으로 서술해야 하며, 명확하고 완결성 있게 작성해야 한다. 
적절한 예시나 개념적 설명을 포함하면서 자연스러운 흐름을 유지해야 한다.
모든 문장은 주어와 서술어의 호응을 고려하여 문장당 평균 17~25어절이 되도록 작성해야 한다.
지문의 전체 글자 수는 공백을 포함해 최소 1200자, 최대 2200자 분량을 지켜야 한다.
허구적인 사건 및 개념, 가상의 인물을 서술하는 것은 금지한다.
지문의 마지막 문단에서 '결론적으로', '결과적으로'와 같이 결론을 지으며 교훈을 주려는 문구를 사용하지 않아야 한다.

앞의 내용을 바탕으로 {type_passage} 분야에서 {keyword}을 핵심 제재로 활용하여 논리적이고 구조적인 지문을 작성해라.

수정됨
앞의 출제 경향 및 작성 원칙을 바탕으로, 과학 분야에서 '혈액', '순환지혈', '과정혈소판'을 핵심 제재로 활용하여 논리적이고 구조적인 수능 국어 독서 영역 지문을 작성하라.
'''
custom_passage = passage_generate_text(system_prompt,user_prompt)
print(custom_passage) 