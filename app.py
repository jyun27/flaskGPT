import os
import pandas as pd
import openai
import json
from dotenv import load_dotenv
from flask import Flask, render_template, request, Response
from openai import OpenAIError, RateLimitError
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI 

load_dotenv()

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_excel("ReChat_data.xlsx")

@app.route('/')
def index():
    return render_template('index.html')


def generate(messages, model_type):
    def stream():
        try:
            # 에이전트 생성
            agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model=model_type), # 모델 정의
            df,                                            # 데이터프레임
            verbose=True,                                  # 추론과정 출력
            agent_type=AgentType.OPENAI_FUNCTIONS,         # 에이전트 타입
            allow_dangerous_code=True                      # 위험한 코드 실행 허용
         ) # 에이전트를 통해 메시지 처리
            
            response = agent.run(messages) 
            
            full_message = ""
            for chunk in response:
                if not chunk:  # Skip empty chunks
                    continue
                
                try:
                    if isinstance(chunk, str):
                        data = json.loads(chunk)
                        content = data['choices'][0]['delta'].get('content', '')
                    elif isinstance(chunk, int):
                        content = str(chunk)
                    else:
                        content = ""
                    full_message += content
                except (json.JSONDecodeError, TypeError):
                    continue  # Skip invalid JSON chunks or type errors
            yield full_message

        except RateLimitError:
            yield "The server is experiencing a high volume of requests. Please try again later."

    return stream()


@app.route('/gpt4', methods=['POST'])
def gpt4():
    data = request.get_json()
    messages = data.get('messages', [])
    model_type = data.get('model_type')

    assistant_response = generate(messages, model_type)
    return Response(assistant_response, mimetype='text/event-stream')



if __name__ == '__main__':
    app.run(debug=True)
