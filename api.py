# api.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import traceback

# --- Importação da Lógica Principal ---
# Isso agora é seguro, pois o loop de main.py está protegido
# pelo 'if __name__ == "__main__":'
try:
    from main import executar_fluxo_gaia, get_session_history, store
except ImportError as e:
    print("="*50)
    print(f"ERRO: Falha ao importar 'main.py'. Detalhe: {e}")
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
        example="Qual a média de emissão da equipe?"
    )
    session_id: str = Field(
        ..., 
        description="Um identificador único para a sessão de chat, para manter o histórico.",
        example="user_session_abc123"
    )

class ChatResponse(BaseModel):
    """Modelo de saída para o endpoint /chat"""
    answer: str = Field(
        ..., 
        description="A resposta gerada pela Gaia.",
        example="Analisei os dados gerais e a média de emissão da equipe é..."
    )
    session_id: str = Field(
        ..., 
        description="O identificador da sessão, retornado para consistência.",
        example="user_session_abc123"
    )

# =====================================
# ENDPOINTS DA API
# =====================================

@app.get("/", tags=["Status"])
def read_root():
    """
    Endpoint de verificação de status.
    Informa se a API da Gaia está online.
    """
    return {"status": "Gaia API está online 🌿"}

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
        
        # Executa a lógica central do seu chatbot
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
        # Log do erro no console do servidor
        print(f"[API ERRO] Erro crítico ao processar /chat para session_id {request.session_id}: {e}")
        traceback.print_exc()
        
        # Retorna um erro 500 para o cliente
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
        # Converte mensagens para um formato JSON serializável
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