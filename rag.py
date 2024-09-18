import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes

llm = ChatOllama(model='Llama-3-Open-Ko-8B-Q8_0:latest')

# BeautifulSoup : HTML 및 XML 문서를 파싱하고 구문 분석하는 데 사용되는 파이썬 라이브러리. 주로 웹 스크레이핑(웹 페이지에서 데이터 추출) 작업에서 사용되며, 웹 페이지의 구조를 이해하고 필요한 정보를 추출하는 데 유용
loader = WebBaseLoader(
    web_paths=("https://www.aitimes.com/news/articleView.html?idxno=159102"
               , "https://www.aitimes.com/news/articleView.html?idxno=159072"
               , "https://www.aitimes.com/news/articleView.html?idxno=158943"
               ),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "article", # 태그
            attrs={"id": ["article-view-content-div"]}, # 태그의 ID 값들
        )
    ),
)
data = loader.load()

#print(f'type : {type(data)} / len : {len(data)}')
#print(f'data : {data}')
#for d in data:
#    print(f'page_content : {d.page_content}')


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(data)
#print(f'len(splits[0].page_content) : {len(splits[0].page_content)}')
#print(splits)


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs = {'device': 'cpu'}, # 모델이 CPU에서 실행되도록 설정. GPU를 사용할 수 있는 환경이라면 'cuda'로 설정할 수도 있음
    encode_kwargs = {'normalize_embeddings': True}, # 임베딩 정규화. 모든 벡터가 같은 범위의 값을 갖도록 함. 유사도 계산 시 일관성을 높여줌
)

vectorstore = FAISS.from_documents(splits,
                                   embedding = embeddings,
                                  )

# 로컬에 DB 저장
MY_FAISS_INDEX = "MY_FAISS_INDEX"
vectorstore.save_local(MY_FAISS_INDEX)

vectorstore = FAISS.load_local(MY_FAISS_INDEX, 
                               embeddings, 
                               allow_dangerous_deserialization=True # 잠재적으로 위험한 데이터 구조나 객체를 포함할 수 있는 인덱스 파일의 로딩을 허용. 주로 자신이 직접 생성하고 저장한 인덱스 파일을 로드할 때 사용
                               )

#retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}) # 유사도 높은 5문장 추출
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}) # 유사도 높은 5문장 추출

#retrieved_docs = retriever.invoke("라마3")
#print(retrieved_docs)

prompt = hub.pull("rlm/rag-prompt") # https://smith.langchain.com/hub/rlm/rag-prompt

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser() 
)
#output = chain.invoke('퍼플렉시티가 투자받은 금액?')
#output = chain.invoke('메타가 출시한것은?')
#print(output)

# map_chain = {"joke": joke_chain, "poem": poem_chain} # 체인에서 이처럼 사용할 때, 자동으로 RunnableParallel 사용됨
# map_chain = RunnableParallel({"joke": joke_chain, "poem": poem_chain})
#map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)
#map_chain.invoke({"topic": "애플"})

app = FastAPI(
        title = "LangChain Server",
        version="1.0",
        description="A simple API server using LangChain+rag",
        )

add_routes(
        app,
        chain,
        path="/chain"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=8000)
#    uvicorn.run(app,host='localhost',port=8000)
