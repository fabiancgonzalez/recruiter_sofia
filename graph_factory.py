"""
Grafo dinamico de reclutamiento (v2): roles y prompts por request.

Agentes:
  - Supervisor Recruiter (rutea)
  - Reclutador: screening de CVs vs puesto
  - Entrevistador: genera preguntas personalizadas del shortlist aprobado

Flujo:
    Supervisor -> Reclutador -> [PAUSA: humano aprueba shortlist]
        | no (tope MAX_AJUSTES)        | si
        v                              v
     vuelve al Reclutador      Entrevistador -> Supervisor -> FINALIZAR
"""
import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt
import sqlite3

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "openai").strip().lower()

def build_model():
    if PROVIDER == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY es requerido cuando PROVIDER=groq.")
        return ChatOpenAI(
            model=os.getenv("MODEL", "llama-3.3-70b-versatile"),
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
        )
    if PROVIDER == "openai":
        return ChatOpenAI(model=os.getenv("MODEL", "gpt-5-mini"), temperature=0)
    if PROVIDER == "vercel_ai_gateway":
        api_key = (
            os.getenv("VERCEL_AI_GATEWAY_API_KEY")
            or os.getenv("AI_GATEWAY_API_KEY")
            or ""
        ).strip()
        if not api_key:
            raise ValueError(
                "VERCEL_AI_GATEWAY_API_KEY o AI_GATEWAY_API_KEY es requerido "
                "cuando PROVIDER=vercel_ai_gateway."
            )
        return ChatOpenAI(
                model=os.getenv("MODEL", "reflection/beam-free"),
            api_key=api_key,
            base_url="https://ai-gateway.vercel.sh/v1",
            temperature=0,
        )
    raise ValueError(
        f"PROVIDER no admitido: {PROVIDER}. Usá 'groq', 'openai' o "
        "'vercel_ai_gateway'."
    )

model = build_model()
MAX_AJUSTES = int(os.getenv("MAX_AJUSTES", "2"))
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoints_reclutador.sqlite")

class RouteDecision(BaseModel):
    next: Literal["reclutador", "entrevistador", "FINALIZAR"]

def mensajes_para_modelo(messages):
    if PROVIDER == "groq":
        return [message.model_copy(update={"name": None}) for message in messages]
    return messages

def make_supervisor(members: list[str], system: str):
    router = model.with_structured_output(RouteDecision, method="function_calling")
    def supervisor(state: MessagesState) -> dict:
        decision = router.invoke([
            SystemMessage(system + "\n\nMiembros del equipo: " + ", ".join(members)
                          + ". Elegi el proximo miembro o FINALIZAR."),
            *mensajes_para_modelo(state["messages"]),
        ])
        return {"next": decision.next}
    return supervisor

def build_graph(cfg: dict):
    """cfg = {supervisor_prompt, reclutador_rol, reclutador_prompt,
              entrevistador_rol, entrevistador_prompt}"""
    reclutador = create_react_agent(
        model, tools=[], name="reclutador",
        prompt=f"Rol: {cfg['reclutador_rol']}\n\n{cfg['reclutador_prompt']}",
    )
    entrevistador = create_react_agent(
        model, tools=[], name="entrevistador",
        prompt=f"Rol: {cfg['entrevistador_rol']}\n\n{cfg['entrevistador_prompt']}",
    )

    def make_worker(agent, nombre):
        def worker(state: MessagesState) -> dict:
            print(f"[equipo] trabaja: {nombre}")
            result = agent.invoke({"messages": mensajes_para_modelo(state["messages"])})
            return {"messages": [result["messages"][-1]]}
        return worker

    def aprobacion(state: MessagesState) -> Command[Literal["entrevistador", "reclutador"]]:
        decision = interrupt({
            "pregunta": "Aprueba el shortlist? Responde 'si' o indica que ajustar.",
            "shortlist": state["messages"][-1].content,
        })
        if decision.strip().lower() in ("si", "sí", "yes", "s", "ok"):
            return Command(goto="entrevistador", update={"messages": []})
        vueltas = sum(1 for m in state["messages"]
                      if getattr(m, "name", None) == "reclutador")
        if vueltas >= MAX_AJUSTES:
            return Command(goto="entrevistador", update={"messages": []})
        return Command(
            goto="reclutador",
            update={"messages": [HumanMessage(
                content=f"El supervisor humano NO aprobo el shortlist. "
                        f"Ajustalo asi: {decision}")]},
        )

    members = ["reclutador", "entrevistador"]
    g = StateGraph(MessagesState)
    g.add_node("supervisor", make_supervisor(members, cfg["supervisor_prompt"]))
    g.add_node("reclutador", make_worker(reclutador, "reclutador"))
    g.add_node("entrevistador", make_worker(entrevistador, "entrevistador"))
    g.add_node("aprobacion", aprobacion)

    g.add_edge(START, "supervisor")
    g.add_edge("reclutador", "aprobacion")
    g.add_edge("entrevistador", "supervisor")
    g.add_conditional_edges("supervisor",
                            lambda s: s["next"] if s["next"] in members else END)

    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    return g.compile(checkpointer=SqliteSaver(conn))

def mensaje_inicial(data: dict) -> HumanMessage:
    """Arma el mensaje de arranque: puesto + caracteristicas + CVs."""
    cvs = "\n\n".join(f"### CV {i+1}: {c['nombre']}\n{c['texto']}"
                        for i, c in enumerate(data["cvs"]))
    return HumanMessage(content=(
        f"POSICION: {data['puesto']}\n"
        f"CARACTERISTICAS DEL PUESTO: {data['caracteristicas']}\n\n"
        f"CVS RECIBIDOS ({len(data['cvs'])}):\n{cvs}"))
