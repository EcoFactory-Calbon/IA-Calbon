import os
import pymongo
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
from collections import Counter

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")

def get_db_connection():
    client = pymongo.MongoClient(MONGO_URL)
    db = client["dbInterEco"]
    return db

class QueryFormularioArgs(BaseModel):
    numero_cracha: int = Field(..., description="Número do crachá do funcionário para buscar o formulário respondido.")

@tool("query_formulario_funcionario", args_schema=QueryFormularioArgs)
def query_formulario_funcionario(numero_cracha: int) -> dict:
    """
    Busca o formulário respondido por um funcionário específico com base no número do crachá.
    """
    db = get_db_connection()
    try:
        form = db.formulario.find_one({"numero_cracha": numero_cracha})
        if not form:
            return {"status": "error", "message": f"Nenhum formulário encontrado para o crachá {numero_cracha}"}
        
        respostas = form.get("respostas", [])
        id_perguntas = [r["id_pergunta"] for r in respostas]
        perguntas_dict = {p["_id"]: p for p in db.perguntas.find({"_id": {"$in": id_perguntas}})}
        
        resultados = []
        for r in respostas:
            pergunta_info = perguntas_dict.get(r["id_pergunta"])
            if pergunta_info:
                resultados.append({
                    "pergunta": pergunta_info["pergunta"],
                    "categoria": pergunta_info["categoria"],
                    "resposta": r["resposta"]
                })

        return {
            "status": "ok",
            "numero_cracha": numero_cracha,
            "data_resposta": str(form.get("data_resposta")),
            "nivel_emissao": form.get("nivel_emissao"),
            "classificacao_emissao": form.get("classificacao_emissao"),
            "respostas": resultados
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool("query_resumo_geral_formularios")
def query_resumo_geral_formularios() -> dict:
    """
    Busca e resume as respostas de TODOS os formulários enviados,
    agregando a contagem de cada resposta por pergunta.
    Não retorna informações de um usuário específico.
    """
    db = get_db_connection()
    try:
        todos_formularios = list(db.formulario.find({}))
        if not todos_formularios:
            return {"status": "ok", "message": "Nenhum formulário encontrado para resumir."}

        perguntas_dict = {p["_id"]: p for p in db.perguntas.find({})}
        
        resumo_respostas = {}

        for form in todos_formularios:
            for resposta_usuario in form.get("respostas", []):
                id_pergunta = resposta_usuario["id_pergunta"]
                resposta_dada = resposta_usuario["resposta"]

                if id_pergunta not in resumo_respostas:
                    pergunta_info = perguntas_dict.get(id_pergunta)
                    if pergunta_info:
                        resumo_respostas[id_pergunta] = {
                            "pergunta": pergunta_info["pergunta"],
                            "categoria": pergunta_info["categoria"],
                            "respostas": Counter()
                        }
                
                if id_pergunta in resumo_respostas:
                    resumo_respostas[id_pergunta]["respostas"][resposta_dada] += 1
        
        lista_resumo = list(resumo_respostas.values())

        return {
            "status": "ok",
            "total_formularios_analisados": len(todos_formularios),
            "resumo_por_pergunta": lista_resumo
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


TOOLS = [
    query_formulario_funcionario,
    query_resumo_geral_formularios
]