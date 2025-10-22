import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import traceback

try:
    from ia_calbon import executar_fluxo_gaia, get_session_history, store
except ImportError as e:
    print("="*50)
    print(f"ERRO: Falha ao importar 'ia.py'. Detalhe: {e}")
    print("Certifique-se de que a modificação do 'if __name__ == \"__main__\":' foi feita.")
    print("="*50)
    exit(1)


# =====================================
# INICIALIZAÇÃO DA API
# =====================================

app = FastAPI(
    title="Gaia API",
    description="API para interagir com a assistente de sustentabilidade Gaia, utilizando um fluxo de RAG e Agentes.",
    version="1.0.0"
)

# =====================================
# MODELOS DE DADOS (Pydantic)
# =====================================

class ChatRequest(BaseModel):
    """Modelo de entrada para o endpoint /chat"""
    question: str = Field(
        ..., 
        description="A pergunta do usuário para a Gaia.",
        example="Qual a melhor forma de diminuir minha emissão de carbono?"
    )
    session_id: str = Field(
        ..., 
        description="Um identificador único para a sessão de chat, para manter o histórico.",
        example="session_test"
    )

class ChatResponse(BaseModel):
    """Modelo de saída para o endpoint /chat"""
    answer: str = Field(
        ..., 
        description="A resposta gerada pela Gaia.",
        example="Não existe uma melhor forma de diminuir sua emissão, e sim várias. Vamos começar identificando ..."
    )
    session_id: str = Field(
        ..., 
        description="O identificador da sessão, retornado para consistência.",
        example="session_test"
    )

# =====================================
# ENDPOINTS DA API
# =====================================

@app.get("/", tags=["Root"], include_in_schema=False)
def read_root():
    """
    Redireciona automaticamente a raiz (/) para a documentação (/docs).
    """
    return RedirectResponse(url="/docs")

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def handle_chat(request: ChatRequest):
    """
    Recebe uma pergunta do usuário e um ID de sessão.
    Processa a pergunta usando o fluxo completo da Gaia (Roteador, Especialistas, Juiz, Orquestrador)
    e retorna a resposta final.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="O campo 'question' não pode estar vazio.")
    if not request.session_id or not request.session_id.strip():
        raise HTTPException(status_code=400, detail="O campo 'session_id' não pode estar vazio.")

    try:
        print(f"[API] Recebida requisição para session_id: {request.session_id}")
        
        resposta_gaia = executar_fluxo_gaia(
            pergunta_usuario=request.question,
            session_id=request.session_id
        )
        
        print(f"[API] Resposta gerada para session_id: {request.session_id}")
        
        return ChatResponse(
            answer=resposta_gaia,
            session_id=request.session_id
        )
        
    except Exception as e:
        print(f"[API ERRO] Erro crítico ao processar /chat para session_id {request.session_id}: {e}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500, 
            detail=f"Ocorreu um erro interno no servidor ao processar sua pergunta."
        )

@app.get("/history/{session_id}", tags=["Chat"])
def get_chat_history_by_id(session_id: str):
    """
    (Endpoint bônus) Retorna o histórico de uma sessão específica.
    Útil para depuração.
    """
    if session_id not in store:
        raise HTTPException(status_code=404, detail="ID de sessão não encontrado.")
    
    try:
        history = get_session_history(session_id)
        messages = [msg.to_json() for msg in history.messages]
        return {"session_id": session_id, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao recuperar histórico: {e}")


# =====================================
# EXECUÇÃO DA API
# =====================================

if __name__ == "__main__":
    """
    Este bloco permite iniciar a API diretamente executando:
    $ python api.py
    
    Para produção, é recomendado usar o comando uvicorn diretamente:
    $ uvicorn api:app --host 0.0.0.0 --port 8000
    """
    print("Iniciando servidor da API Gaia em http://127.0.0.1:8000 ...")
    print("Acesse http://127.0.0.1:8000/docs para ver a documentação interativa (Swagger).")
    uvicorn.run(app, host="127.0.0.1", port=8000)