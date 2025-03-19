import os
import sys
from dotenv import load_dotenv 
from langchain_community.document_loaders import DirectoryLoader, TextLoader  
from langchain_openai import OpenAIEmbeddings  
from langchain_community.vectorstores import FAISS  

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()
API_KEY = os.getenv("API_KEY")
os.environ['OPENAI_API_KEY'] = API_KEY

# 디렉토리 내의 모든 문제 관련 .txt 파일을 불러오기
loader_question = DirectoryLoader("./문항생성/question_doc", glob="*.txt", 
                                  loader_cls=lambda path: TextLoader(path, encoding="utf-8"))
documents_question = loader_question.load()

# OpenAI 임베딩 모델 초기화
embeddings_model = OpenAIEmbeddings()

# 전체 문서 FAISS 인덱스 생성
texts_question = [doc.page_content for doc in documents_question]
metadata_question = [doc.metadata for doc in documents_question]
vector_db_question_all = FAISS.from_texts(texts_question, embeddings_model, metadatas=metadata_question)

# FAISS 인덱스 저장 (문항 전체)
faiss_question_all_path = "./문항생성/faiss_index/faiss_index_question_all"

vector_db_question_all.save_local(faiss_question_all_path)
print(f"문제 전체 문서 FAISS 벡터 DB 저장 완료: {faiss_question_all_path}")