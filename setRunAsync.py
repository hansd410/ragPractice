#kernel should be chosen to python3.8 AzureML
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langserve import add_routes

#llm = ChatOllama(model='llama3:latest')
llm = ChatOllama(model='Llama-3-Open-Ko-8B-Q8_0:latest',disable_streaming=True)
#llm.invoke("hello world")

#prompt = ChatPromptTemplate.from_messages([
#    ("system", "You are a helpful, professional assistant named 권봇. Introduce yourself first, and answer the questions. answer me in Korean no matter what. "),
#    ("user", "{input}")
#])

#chain = prompt | llm | StrOutputParser()
#chain.invoke({"input": "What is stock?"})

joke_chain = (
    ChatPromptTemplate.from_template("{topic}에 관련해서 짧은 농담 말해줘") 
    | llm 
    | StrOutputParser()
    )
poem_chain = (
    ChatPromptTemplate.from_template("{topic}에 관련해서 시 2줄 써줘") 
    | llm
    | StrOutputParser()
    )

# map_chain = {"joke": joke_chain, "poem": poem_chain} # 체인에서 이처럼 사용할 때, 자동으로 RunnableParallel 사용됨
# map_chain = RunnableParallel({"joke": joke_chain, "poem": poem_chain})
map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

#map_chain.invoke({"topic": "애플"})


app = FastAPI(
        title = "LangChain Server",
        version="1.0",
        description="A simple API server using LangChain",
        )

add_routes(
        app,
        map_chain,
        path="/chain"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=8000)
#    uvicorn.run(app,host='localhost',port=8000)
