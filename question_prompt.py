
import os
import numpy as np
import json
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain, LLMChain, StuffDocumentsChain
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.docstore.document import Document
import requests

# .env 파일 로드
load_dotenv()

# API를 통해 데이터 가져오기
def fetch_data_from_api(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as api_error:
        print(f"API 요청 오류: {api_error}")
        return None

api_url = "https://example.com/api/data"  # 실제 API URL 입력 필요
api_response = fetch_data_from_api(api_url)

if api_response:
    question_type = api_response.get("question_type", "기본 질문 유형")
    type_question = api_response.get("type_question", "기본 유형")
    custom_passage = api_response.get("custom_passage", "기본 지문")
else:
    question_type = "기본 질문 유형"
    type_question = "기본 유형"
    custom_passage = "기본 지문"

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
    return response.choices[0].message.content.strip()

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
vector_db_question_all = FAISS.load_local(faiss_question_all_path, embeddings_model)

# 유사한 문서 검색
question_guidelines = vector_db_question_all.similarity_search(type_question, k=1)

# 검색 결과 출력
for i, res in enumerate(question_guidelines):
    print(f"\n[{i+1}] 검색된 문서:")
    print("-" * 50)
    print(res.page_content)
    print("-" * 50)
    print("📄 Metadata:", res.metadata)

q_user_prompt = f"""

다음은 {question_type} 유형의 문항과 선지를 작성할 때 반드시 고려해야 할 기준이다.

{question_guidelines}

다음으로 제시하는 지문과 문항 유형을 바탕으로 5개의 선지로 이루어진 문항 1개를 작성해라.
[지문]
{custom_passage}
[문항 유형]
{type_question}
문항을 출력하는 형식은 아래와 같다. 
정답 및 해설은 선지 번호(1, 2, 3, 4, 5)를 활용하여 정답 선지의 근거와 오답의 틀린 이유를 포함한 상세 해설을 최소 100자, 최대 200자로 출력해야 한다. 

    {
    "generated_question": "질문",
    "generated_option": ["선지1", "선지2", "선지3", "선지4", "선지5"],
    "generated_answer": "정답",
    "generated_description": "해설"
    }

"""

question_data = generate_text(q_system_text, q_user_prompt)
question_data
