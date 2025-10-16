# main.py

from dotenv import load_dotenv
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Importando sua lista de ferramentas do arquivo tools.py
from tools import TOOLS

# =====================================
# BASE
# =====================================
store = {}
TZ = ZoneInfo("America/Sao_Paulo")
today = datetime.now(TZ).date()

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

load_dotenv()

# =====================================
# MODELOS LLM
# =====================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    top_p=0.95,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

llm_fast = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# =====================================
# FUNÇÃO DE BADWORDS
# =====================================
def carregar_badwords(caminho="badwords.txt"):
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return [linha.strip().lower() for linha in f if linha.strip()]
    except FileNotFoundError:
        print(f"Aviso: Arquivo '{caminho}' não encontrado. Nenhuma badword carregada.")
        return []

BAD_WORDS = carregar_badwords()

# =====================================
# PROMPTS DE NÍVEL PROFISSIONAL (COM CORREÇÃO)
# =====================================

# 1. Roteador: O Porteiro Inteligente
system_prompt_roteador = """
### PAPEL
Você é o "Porteiro de Intenções" da Gaia. Sua única e exclusiva função é analisar a pergunta do usuário e encaminhá-la para o especialista correto, de forma rápida e precisa. Hoje é {today}.

### CATEGORIAS DE TRIAGEM
1.  `diagnostico`: Solicitações que exigem consulta a dados ou análise. Exemplos: "Analise meu formulário", "Qual a média da empresa?", "Quais os dados do crachá 123?".
2.  `carbono`: Perguntas gerais sobre sustentabilidade, CO₂, ou funcionalidades do aplicativo que não exigem acesso a dados. Exemplos: "O que é pegada de carbono?", "Dicas para reduzir CO₂".
3.  `fora_de_escopo`: Qualquer outra coisa.

### FORMATO DE SAÍDA OBRIGATÓRIO
ROUTE=<diagnostico|carbono|fora_de_escopo>
PERGUNTA_ORIGINAL=<mensagem completa do usuário>
"""
prompt_roteador = ChatPromptTemplate.from_messages([
    ("system", system_prompt_roteador),
    ("human", "{input}")
])

# 2. Especialista de Carbono: O Sábio Focado
system_prompt_carbono = """
### PAPEL
Você é o "Sábio da Sustentabilidade" da Gaia. Sua especialidade é fornecer informações claras e factuais sobre sustentabilidade e emissões de carbono.

### DIRETRIZES
- Você NÃO tem acesso a dados de usuários ou ferramentas. Sua base é o conhecimento geral.
- Se a pergunta exigir uma análise de dados, afirme que essa análise deve ser feita pelo especialista de diagnóstico.
- Sua resposta final DEVE ser um único e válido objeto JSON.

### ESQUEMA DE SAÍDA (JSON OBRIGATÓRIO)
{{
  "dominio": "carbono",
  "intencao": "informar",
  "resposta": "Sua resposta clara e direta à pergunta do usuário.",
  "recomendacao": "Uma dica prática relacionada ao tópico, ou uma string vazia se não aplicável."
}}
""" # <<< CORREÇÃO APLICADA AQUI com chaves duplas
prompt_carbono = ChatPromptTemplate.from_messages([
    ("system", system_prompt_carbono),
    ("human", "{input}"),
])

# 3. Agente de Diagnóstico: O Analista Sênior
system_prompt_diag = """
### PAPEL
Você é um Analista de Dados Sênior, especialista em sustentabilidade. Sua missão é transformar dados brutos de formulários em insights acionáveis.

### FLUXO DE TRABALHO (ALGORITMO OBRIGATÓRIO)
1.  **Análise da Solicitação**: Identifique se a pergunta do usuário é sobre um indivíduo (requer `numero_cracha`) ou sobre o coletivo (resumo geral).
2.  **Seleção e Execução da Ferramenta**:
    - Para análise individual com crachá, chame `query_formulario_funcionario`.
    - Para análise geral/coletiva, chame `query_resumo_geral_formularios`.
3.  **Análise Crítica do Resultado**: Analise CUIDADOSAMENTE os dados retornados pela ferramenta.
4.  **Geração do JSON Final**: Com base na sua análise, construa o objeto JSON de saída. Sua tarefa SÓ termina quando este JSON for gerado.

### MANUSEIO DE ERROS
- Se a ferramenta retornar um erro (ex: "Nenhum formulário encontrado"), o campo `resposta` do seu JSON deve explicar o problema de forma amigável e o campo `recomendacao` deve sugerir uma ação (ex: "Por favor, verifique se o número do crachá está correto.").

### ESQUEMA DE SAÍDA (JSON OBRIGATÓRIO)
{{
  "dominio": "diagnostico",
  "intencao": "analisar",
  "resposta": "Sua análise detalhada e conclusões baseadas nos dados da ferramenta. Em caso de erro, a explicação amigável do erro.",
  "recomendacao": "Uma sugestão prática e acionável baseada nos dados. Em caso de erro, uma sugestão para resolver o problema."
}}
""" # <<< CORREÇÃO APLICADA AQUI com chaves duplas
prompt_diag = ChatPromptTemplate.from_messages([
    ("system", system_prompt_diag),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 4. Orquestrador: A Voz Empática da Gaia
system_prompt_orq = """
### PAPEL
Você é a "Voz de Gaia". Sua função é receber um objeto JSON técnico de um especialista e traduzi-lo em uma mensagem final para o usuário que seja calorosa, empática e motivadora.

### ENTRADA
Você receberá um objeto JSON no campo `{input}` com os campos "resposta" e "recomendacao".

### REGRAS
- Sintetize a "resposta" em um parágrafo amigável.
- Apresente a "recomendacao" como uma "Sugestão" prática.
- Use emojis leves e apropriados (🌿, ✨, 💡).
- NÃO invente informações. Baseie-se estritamente no JSON recebido.

### FORMATO DE SAÍDA
<Sua resposta amigável e elaborada>

Sugestão: <Sua apresentação da recomendação de forma clara e motivadora>
"""
prompt_orq = ChatPromptTemplate.from_messages([
    ("system", system_prompt_orq),
    ("human", "{input}")
])

# =====================================
# AGENTES E CADEIAS
# =====================================
carbono_chain = prompt_carbono | llm_fast | StrOutputParser()

diag_agente = create_tool_calling_agent(llm, TOOLS, prompt_diag)
diag_exec = AgentExecutor(agent=diag_agente, tools=TOOLS, verbose=True)
diag_chain = RunnableWithMessageHistory(
    diag_exec,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

router_chain = prompt_roteador | llm_fast | StrOutputParser()
orquestrador_chain = prompt_orq | llm_fast | StrOutputParser()

# =====================================
# EXECUÇÃO DO FLUXO
# =====================================
def executar_fluxo_gaia(pergunta_usuario: str, session_id: str):
    if any(word in pergunta_usuario.lower() for word in BAD_WORDS):
        return "Por favor, vamos manter a conversa respeitosa e focada em sustentabilidade. 🌿"

    chat_history = get_session_history(session_id)

    resposta_roteador = router_chain.invoke({
        "input": pergunta_usuario,
        "today": today.isoformat()
    }).strip()
    print(f"[DEBUG] Roteador retornou:\n{resposta_roteador}\n")

    route_info = {line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
                  for line in resposta_roteador.split("\n") if "=" in line}
    route = route_info.get("ROUTE", "fora_de_escopo")
    especialista_input = route_info.get("PERGUNTA_ORIGINAL", pergunta_usuario)

    resposta_final = ""
    if route == "carbono":
        print(f"[DEBUG] Chamando especialista CARBONO...")
        json_especialista = carbono_chain.invoke({"input": especialista_input})
        print(f"[DEBUG] Especialista CARBONO respondeu:\n{json_especialista}")
        resposta_final = orquestrador_chain.invoke({"input": json_especialista})

    elif route == "diagnostico":
        print(f"[DEBUG] Chamando AGENTE DE DIAGNÓSTICO...")
        resposta_agente = diag_chain.invoke(
            {"input": especialista_input},
            config={"configurable": {"session_id": session_id}}
        )
        json_analisado = resposta_agente['output']
        print(f"[DEBUG] Agente DIAGNÓSTICO respondeu:\n{json_analisado}")
        resposta_final = orquestrador_chain.invoke({"input": json_analisado})

    else:
        resposta_final = "Olá! Sou a Gaia. Meu foco é ajudar com a redução de emissões de carbono. Como posso te apoiar nesse tema hoje? 🌿"

    chat_history.add_user_message(pergunta_usuario)
    chat_history.add_ai_message(resposta_final)
    return resposta_final

# =====================================
# LOOP INTERATIVO
# =====================================
print("🌿 Gaia iniciada (versão profissional). Diga 'sair' para encerrar.\n")
SESSION_ID = "sessao_unica"

while True:
    user_input = input("> ")
    if user_input.lower() in ["sair", "exit", "quit", "fim", "tchau"]:
        print("\nGaia: Até logo! 🌿")
        break
    try:
        resposta = executar_fluxo_gaia(user_input, session_id=SESSION_ID)
        print(f"\nGaia: {resposta}\n")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")