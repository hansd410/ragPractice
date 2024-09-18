import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

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
    model_kwargs = {'device': 'cuda'}, # 모델이 CPU에서 실행되도록 설정. GPU를 사용할 수 있는 환경이라면 'cuda'로 설정할 수도 있음
    encode_kwargs = {'normalize_embeddings': True}, # 임베딩 정규화. 모든 벡터가 같은 범위의 값을 갖도록 함. 유사도 계산 시 일관성을 높여줌
)
embed = embeddings.embed_documents(
    [
        "안녕 영광",
        "동해물과 백두산",
        "마르고 닳도록",
        "하느님이 보우하사",
        "우리나라 만세"
    ]
)
print(f"len(embed): {len(embed)}")
print(f"len(embed[0]): {len(embed[0])}")
print(f"len(embed[1]): {len(embed[1])}")
print(f"embed: {embed}")
