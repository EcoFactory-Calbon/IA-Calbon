# main.py

from dotenv import load_dotenv
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
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
# PROMPTS
# =====================================

example_prompt_base = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}"),
])

# 1. Roteador
PERSONA_SISTEMA_GAIA = """Você é a Gaia — uma assistente IA especialista em sustentabilidade e análise de dados de carbono. Você é objetiva, confiável e empática, com foco em ajudar o usuário a entender e reduzir seu impacto ambiental.
- Evite jargões.
"""

system_prompt_roteador = ("system",
    """
### PERSONA SISTEMA
{persona}

### PAPEL
- Acolher o usuário e manter o foco em SUSTENTABILIDADE.
- Decidir a rota: {{diagnostico | carbono}} ou se a pergunta é fora escopo/saudação.
- Responder DIRETAMENTE em texto puro para:
  (a) `saudacao`: saudações/small talk.
  (b) `fora_de_escopo`: redirecionando para sustentabilidade.
- Usar o PROTOCOLO DE ENCAMINHAMENTO para:
  (a) `diagnostico`: Análise de dados, formulários, crachás.
  (b) `carbono`: Dicas, conceitos de CO2, pegada de carbono (sem dados).

### REGRAS
- Para `saudacao` e `fora_de_escopo`, SEJA AMIGÁVEL e VARIE a resposta.
- Para `diagnostico` ou `carbono`, NÃO responda ao usuário, use o protocolo.

### PROTOCOLO DE ENCAMINHAMENTO (Texto puro)
ROUTE=<diagnostico|carbono>
PERGUNTA_ORIGINAL=<mensagem completa do usuário, sem edições>
PERSONA=<copie a PERSONA SISTEMA daqui>

### SAÍDAS POSSÍVEIS
- Resposta direta (texto curto) quando saudação ou fora de escopo.
- Encaminhamento ao especialista (diagnostico/carbono) usando o protocolo.

### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

shots_roteador = [
    {
        "human": "Oi, tudo bem?",
        "ai": "Olá! Sou a Gaia 🌿. Prontos para analisar alguns dados de sustentabilidade hoje?"
    },
    {
        "human": "Qual a previsão do tempo?",
        "ai": "Meu foco é 100% em sustentabilidade. Posso ajudar analisando a média de emissão da equipe ou dando dicas de CO2. O que prefere?"
    },
    {
        "human": "Qual a média de emissão do time?",
        "ai": f"ROUTE=diagnostico\nPERGUNTA_ORIGINAL=Qual a média de emissão do meu formulário?\nPERSONA={PERSONA_SISTEMA_GAIA}"
    },
    {
        "human": "O que é pegada de carbono?",
        "ai": f"ROUTE=carbono\nPERGUNTA_ORIGINAL=O que é pegada de carbono?\nPERSONA={PERSONA_SISTEMA_GAIA}"
    },
    {
        "human": "Me dá os dados do crachá 123.",
        "ai": f"ROUTE=diagnostico\nPERGUNTA_ORIGINAL=Me dá os dados do crachá 123.\nPERSONA={PERSONA_SISTEMA_GAIA}"
    },
    {
        "human": "Bom dia",
        "ai": "Bom dia! 💡 Sobre o que vamos conversar hoje: análise de dados ou dicas de sustentabilidade?"
    }
]

fewshots_roteador = FewShotChatMessagePromptTemplate(
    examples=shots_roteador,
    example_prompt=example_prompt_base
)

prompt_roteador = ChatPromptTemplate.from_messages([
    system_prompt_roteador,
    fewshots_roteador,
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
]).partial(persona=PERSONA_SISTEMA_GAIA)

# 2. Especialista de Carbono
system_prompt_carbono = ("system",
"""
### PAPEL
Você é o "Sábio da Sustentabilidade" da Gaia. Sua especialidade é fornecer informações claras e factuais sobre sustentabilidade e emissões de carbono. Sua base é o conhecimento geral.

### CONTEXTO
- Você recebe uma `PERGUNTA_ORIGINAL` do roteador.
- Se a pergunta exigir uma análise de dados, afirme que essa análise deve ser feita pelo especialista de diagnóstico.

### REGRAS
- Sua resposta final DEVE ser um único e válido objeto JSON.

### ESQUEMA DE SAÍDA (JSON OBRIGATÓRIO)
{{
  "dominio": "carbono",
  "intencao": "informar",
  "resposta": "Sua resposta clara e direta à pergunta do usuário.",
  "recomendacao": "Uma dica prática relacionada ao tópico, ou uma string vazia se não aplicável."
}}

### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

shots_carbono = [
    {
        "human": "O que é pegada de carbono?",
        "ai": """{
  "dominio": "carbono",
  "intencao": "informar",
  "resposta": "Pegada de carbono é o volume total de gases de efeito estufa (GEE) gerados por nossas atividades diárias, medido em toneladas de CO2.",
  "recomendacao": "Pequenas ações, como reduzir o consumo de carne ou usar menos o carro, ajudam a diminuí-la."
}"""
    },
    {
        "human": "Como posso reduzir minha emissão em casa?",
        "ai": """{
  "dominio": "carbono",
  "intencao": "informar",
  "resposta": "Em casa, o foco é reduzir o consumo de energia e a produção de lixo.",
  "recomendacao": "Tente trocar lâmpadas comuns por LED, desligar aparelhos da tomada e separar seu lixo orgânico para compostagem."
}"""
    },
    {
        "human": "Qual a média de emissão da minha equipe?", 
        "ai": """{
  "dominio": "carbono",
  "intencao": "informar",
  "resposta": "Eu forneço informações gerais sobre sustentabilidade. Para analisar dados específicos da sua equipe, você precisa falar com nosso especialista em diagnóstico.",
  "recomendacao": "Tente perguntar 'qual a média de emissão da empresa?' para o especialista correto."
}"""
    }
]

fewshots_carbono = FewShotChatMessagePromptTemplate(
    examples=shots_carbono,
    example_prompt=example_prompt_base,
)

prompt_carbono = ChatPromptTemplate.from_messages([
    system_prompt_carbono,
    MessagesPlaceholder(variable_name="chat_history"),
    fewshots_carbono,
    ("human", "{input}"), 
])

# 3. Agente de Diagnóstico
system_prompt_diag = ("system",
"""
### PAPEL
Você é um Analista de Dados Sênior, especialista em sustentabilidade. Sua missão é transformar dados brutos de formulários em insights acionáveis usando as ferramentas disponíveis.

### CONTEXTO
- Você recebe uma `PERGUNTA_ORIGINAL` do roteador.
- Hoje é {today}

### FLUXO DE TRABALHO (ALGORITMO OBRIGATÓRIO)
1.  **Análise da Solicitação**: Identifique se a pergunta é sobre um indivíduo (`numero_cracha`) ou coletivo.
2.  **Seleção e Execução da Ferramenta**:
    - Para análise individual, chame `query_formulario_funcionario`.
    - Para análise geral/coletiva, chame `query_resumo_geral_formularios`.
3.  **Análise Crítica do Resultado**: Analise CUIDADOSAMENTE os dados retornados pela ferramenta.
4.  **Geração do JSON Final**: Com base na sua análise, construa o objeto JSON de saída.

### MANUSEIO DE ERROS
- Se a ferramenta retornar um erro (ex: "Nenhum formulário encontrado"), o campo `resposta` do seu JSON deve explicar o problema de forma amigável.

### ESQUEMA DE SAÍDA (JSON OBRIGATÓRIO)
{{
  "dominio": "diagnostico",
  "intencao": "analisar",
  "resposta": "Sua análise detalhada e conclusões baseadas nos dados da ferramenta. Em caso de erro, a explicação amigável do erro.",
  "recomendacao": "Uma sugestão prática e acionável baseada nos dados. Em caso de erro, uma sugestão para resolver o problema."
}}

### HISTÓRICO DA CONVERSA
{chat_history}
### HISTÓRICO INTERNO DO AGENTE (Não mexa)
{agent_scratchpad}
"""
)


prompt_diag = ChatPromptTemplate.from_messages([
    system_prompt_diag, 
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
]).partial(today=today.isoformat())


# 4. Orquestrador

system_prompt_orq = ("system",
"""
### PAPEL
Você é a "Voz de Gaia". Sua função é receber um objeto JSON técnico de um especialista e traduzi-lo em uma mensagem final para o usuário que seja calorosa, empática e motivadora.

### ENTRADA
Você receberá um objeto JSON técnico no campo `{input}`.

### REGRAS CRÍTICAS
- **Analise o `chat_history`. Se a conversa já começou (ou seja, se `chat_history` não estiver vazio), NÃO use saudações como "Olá!", "Oi!", etc. Vá direto ao ponto.**
- Sintetize a "resposta" do JSON em um parágrafo amigável.
- Apresente a "recomendacao" como uma "Sugestão" prática.
- Use emojis leves e apropriados (🌿, ✨, 💡).
- NÃO invente informações. Baseie-se estritamente no JSON recebido.

### FORMATO DE SAÍDA
<Sua resposta amigável e elaborada>

Sugestão: <Sua apresentação da recomendação de forma clara e motivadora>

### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

shots_orquestrador = [
    {
        "human": """{
  "dominio": "diagnostico",
  "intencao": "analisar",
  "resposta": "Análise do crachá 123: Emissão de 80.5 kgCO2/mês. Principal fonte: Carro a Gasolina (40km/dia).",
  "recomendacao": "Considere usar o transporte fretado da empresa 2x por semana."
}""",
        "ai": """Analisei os dados do crachá 123 e a emissão mensal é de 80.5 kgCO2 💡. A principal fonte é o deslocamento com carro a gasolina, rodando cerca de 40km por dia.

Sugestão: Que tal experimentar o transporte fretado da empresa duas vezes por semana? Isso já faria uma grande diferença! 🌿"""
    },
    {
        "human": """{
  "dominio": "carbono",
  "intencao": "informar",
  "resposta": "Pegada de carbono é o volume total de gases de efeito estufa (GEE) gerados por nossas atividades.",
  "recomendacao": "Pequenas ações, como reduzir o consumo de carne, ajudam a diminuí-la."
}""",
        "ai": """A pegada de carbono é basicamente o volume total de gases de efeito estufa que geramos em nossas atividades do dia a dia. ✨

Sugestão: Pequenas ações já ajudam muito, como diminuir um pouco o consumo de carne vermelha durante a semana! 🌿"""
    },
    {
        "human": """{
  "dominio": "diagnostico",
  "intencao": "analisar",
  "resposta": "Erro: Nenhum formulário encontrado para o crachá 999.",
  "recomendacao": "Por favor, verifique se o número do crachá está correto."
}""",
        "ai": """Hmm, parece que não encontrei nenhum formulário preenchido para o crachá 999.

Sugestão: Você poderia verificar se o número está digitado corretamente, por favor? 💡"""
    }
]

fewshots_orquestrador = FewShotChatMessagePromptTemplate(
    examples=shots_orquestrador,
    example_prompt=example_prompt_base,
)

prompt_orq = ChatPromptTemplate.from_messages([
    system_prompt_orq,
    MessagesPlaceholder(variable_name="chat_history"),
    fewshots_orquestrador,
    ("human", "{input}")
])

# =====================================
# AGENTES E CADEIAS
# =====================================
# Roteador
router_chain = RunnableWithMessageHistory(
    prompt_roteador | llm_fast | StrOutputParser(),
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Especialista Carbono
carbono_chain = RunnableWithMessageHistory(
    prompt_carbono | llm_fast | StrOutputParser(),
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Agente Diagnóstico
diag_agente = create_tool_calling_agent(llm, TOOLS, prompt_diag)
diag_exec = AgentExecutor(agent=diag_agente, tools=TOOLS, verbose=True)
diag_chain = RunnableWithMessageHistory(
    diag_exec,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Orquestrador 
orquestrador_chain = RunnableWithMessageHistory(
    prompt_orq | llm_fast | StrOutputParser(),
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# =====================================
# EXECUÇÃO DO FLUXO
# =====================================

def executar_fluxo_gaia(pergunta_usuario: str, session_id: str):
    if any(word in pergunta_usuario.lower() for word in BAD_WORDS):
        return "Por favor, vamos manter a conversa respeitosa e focada em sustentabilidade. 🌿"

    config = {"configurable": {"session_id": session_id}}
    chat_history = get_session_history(session_id)

    resposta_roteador = router_chain.invoke({"input": pergunta_usuario}, config=config).strip()
    print(f"[DEBUG] Roteador retornou:\n{resposta_roteador}\n")

    resposta_final = ""

    if not resposta_roteador.startswith("ROUTE="):
        print("[DEBUG] Rota de Resposta Direta (Saudação/Fora de Escopo).")
        resposta_final = resposta_roteador
    
    else:
        print("[DEBUG] Rota de Especialista (Diagnóstico/Carbono).")
        
        route_info = {}
        for line in resposta_roteador.split("\n"):
            if "=" in line:
                partes = line.split("=", 1)
                if len(partes) == 2:
                    route_info[partes[0].strip()] = partes[1].strip()

        route = route_info.get("ROUTE", "fora_de_escopo")
        especialista_input = route_info.get("PERGUNTA_ORIGINAL", pergunta_usuario)

        json_especialista = ""
        
        if route == "carbono":
            print(f"[DEBUG] Chamando especialista CARBONO...")
            json_especialista = carbono_chain.invoke(
                {"input": especialista_input},
                config=config
            )
            print(f"[DEBUG] Especialista CARBONO respondeu:\n{json_especialista}")

        elif route == "diagnostico":
            print(f"[DEBUG] Chamando AGENTE DE DIAGNÓSTICO...")
            resposta_agente = diag_chain.invoke(
                {"input": especialista_input},
                config=config
            )
            json_especialista = resposta_agente['output']
            print(f"[DEBUG] Agente DIAGNÓSTICO respondeu:\n{json_especialista}")
        
        else:
            print(f"[DEBUG] Rota '{route}' inesperada no protocolo. Usando fallback.")
            resposta_final = "Sou a Gaia e meu dever é ajudar com sustentabilidade. Como posso te ajudar com isso? 🌿"

        if json_especialista and not resposta_final:
            resposta_final = orquestrador_chain.invoke(
                {"input": json_especialista},
                config=config
            )
    chat_history.add_user_message(pergunta_usuario)
    chat_history.add_ai_message(resposta_final)
    
    return resposta_final

# =====================================
# LOOP INTERATIVO
# =====================================

print("🌿 Gaia iniciada (versão profissional com Few-Shots). Diga 'sair' para encerrar.\n")
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