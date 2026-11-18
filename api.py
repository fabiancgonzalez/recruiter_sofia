"""
API del sistema reclutador multiagente (v2).

POST /procesos              {"puesto", "caracteristicas", "cvs": [{nombre, texto}],
                             "supervisor_rol", "supervisor_prompt",
                             "reclutador_rol", "reclutador_prompt",
                             "entrevistador_rol", "entrevistador_prompt"} -> pid
POST /cvs/pdf               multipart: files=[.pdf...] -> [{"nombre", "texto"}]
GET  /procesos/{pid}        estado + shortlist/preguntas cuando esta en pausa
POST /procesos/{pid}/respuesta  {"respuesta": "si" | "ajustes..."}
"""
import os, io, asyncio, uuid, logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AuthenticationError, NotFoundError, PermissionDeniedError, RateLimitError

from langchain_core.messages import AIMessage
from langgraph.types import Command
from graph_factory import PROVIDER, build_graph, mensaje_inicial, model

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.getLogger("uvicorn.error").info(
        "Proveedor del modelo: %s; modelo: %s", PROVIDER, model.model_name)
    yield

app = FastAPI(title="Reclutador multiagente", version="2.0", lifespan=lifespan)
PROCESOS: dict = {}
log = logging.getLogger("reclutador")


def _run_background(graph, *args, proc=None):
    """Ejecuta graph.invoke en un hilo sin bloquear la respuesta HTTP."""
    if proc is not None:
        proc["running"] = True
        proc.pop("error", None)

    def _on_done(fut):
        exc = fut.exception()
        if exc:
            log.error("Error en ejecucion en segundo plano", exc_info=exc)
            if proc is not None:
                proveedor = {
                    "groq": "Groq",
                    "vercel_ai_gateway": "Vercel AI Gateway",
                }.get(PROVIDER, "OpenAI")
                if isinstance(exc, RateLimitError):
                    if "credits" in str(exc).lower() or "insufficient_quota" in str(exc).lower():
                        proc["error"] = (f"{proveedor} no tiene créditos disponibles. Revisá la facturación "
                                         "de la cuenta y volvé a iniciar el proceso.")
                    else:
                        proc["error"] = f"{proveedor} alcanzó el límite de solicitudes. Esperá y volvé a intentar."
                elif isinstance(exc, AuthenticationError):
                    proc["error"] = f"{proveedor} rechazó la clave de API. Revisá la configuración del servidor."
                elif isinstance(exc, PermissionDeniedError):
                    if PROVIDER == "vercel_ai_gateway" and "free tier" in str(exc).lower():
                        proc["error"] = (
                            f"El modelo {model.model_name} no está disponible en el nivel gratuito "
                            "de Vercel AI Gateway. Configurá MODEL con un modelo gratuito, por "
                            "ejemplo reflection/beam-free, o agregá créditos."
                        )
                    else:
                        proc["error"] = (
                            f"{proveedor} denegó el acceso. Revisá los permisos de la cuenta y del modelo."
                        )
                elif isinstance(exc, NotFoundError):
                    host = exc.response.request.url.host
                    if host == "api.groq.com":
                        proveedor = "Groq"
                    elif host == "api.openai.com":
                        proveedor = "OpenAI"
                    else:
                        proveedor = "El servidor configurado"
                    proc["error"] = (
                        f"{proveedor} no encontró el modelo {model.model_name} o la cuenta no tiene acceso. "
                        "Revisá PROVIDER, MODEL y los modelos disponibles para la cuenta de la clave configurada."
                    )
                else:
                    proc["error"] = "No se pudo completar el proceso. Revisá el registro del servidor."
        if proc is not None:
            proc["running"] = False
    fut = asyncio.get_running_loop().run_in_executor(
        None, lambda: graph.invoke(*args))
    fut.add_done_callback(_on_done)

class CV(BaseModel):
    nombre: str
    texto: str

class NuevoProceso(BaseModel):
    puesto: str
    caracteristicas: str
    cvs: list[CV]
    supervisor_rol: str = "Supervisor Recruiter"
    supervisor_prompt: str = ("Coordinas el proceso de seleccion. Delega el screening "
                              "al reclutador; cuando el shortlist este aprobado por el "
                              "humano, delega en el entrevistador. Cuando las preguntas "
                              "de entrevista esten listas, FINALIZAR.")
    reclutador_rol: str = "Recruiter IT Senior"
    reclutador_prompt: str = ("Sos un recruiter experto. Recibis el puesto con sus "
                              "caracteristicas y una lista de CVs. Para cada CV evalua: "
                              "match de habilidades tecnicas, experiencia relevante, "
                              "seniority y expectativa salarial si figura. Entrega un "
                              "ranking de mejor a peor con puntaje 1-10, justificacion "
                              "corta por candidato, y una shortlist de los 3 mejores "
                              "con puntos a preguntar en entrevista. Se objetivo: si un "
                              "dato no esta en el CV, no lo asumas.")
    entrevistador_rol: str = "Entrevistador Tecnico"
    entrevistador_prompt: str = ("Sos un entrevistador tecnico. Con el shortlist "
                                 "aprobado y los CVs originales, genera para cada "
                                 "candidato seleccionado: 5 preguntas tecnicas "
                                 "personalizadas (basadas en su stack y experiencia), "
                                 "3 preguntas situacionales/soft skills y 2 red flags "
                                 "a verificar. Organiza por candidato.")

class Respuesta(BaseModel):
    respuesta: str

# ---------------- PDF -> texto ----------------
@app.post("/cvs/pdf")
async def pdf_a_texto(files: list[UploadFile] = File(...)):
    """Recibe PDFs y devuelve el texto extraido (para precargar en el panel)."""
    from pypdf import PdfReader
    resultados = []
    for f in files:
        try:
            reader = PdfReader(io.BytesIO(await f.read()))
            texto = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            raise HTTPException(400, f"No se pudo leer el PDF: {f.filename}")
        if not texto.strip():
            raise HTTPException(400, f"El PDF no tiene texto extraible (escaneado?): {f.filename}")
        resultados.append({"nombre": f.filename.rsplit(".", 1)[0], "texto": texto})
    return {"cvs": resultados}

# ---------------- procesos ----------------
def _snapshot(graph, config) -> dict:
    estado = graph.get_state(config)
    mensajes = estado.values.get("messages", [])
    pendiente = bool(estado.next and "aprobacion" in str(estado.next))
    resultado = mensajes[-1].content if mensajes and isinstance(mensajes[-1], AIMessage) else None
    terminado_sin_resultado = not estado.next and not resultado
    return {
        "estado": "en_pausa" if estado.next else "error" if terminado_sin_resultado else "finalizado",
        "nodo_actual": list(estado.next),
        "total_mensajes": len(mensajes),
        "pendiente_aprobacion": pendiente,
        "shortlist": (mensajes[-1].content if pendiente and mensajes else None),
        "resultado": resultado if not estado.next else None,
        **({"error": "El proceso terminó sin generar un resultado. Revisá la configuración y volvé a intentar."}
           if terminado_sin_resultado else {}),
    }

@app.get("/")
def home():
    return RedirectResponse("/panel")

@app.post("/procesos")
async def crear_proceso(body: NuevoProceso):
    if not body.cvs:
        raise HTTPException(400, "Carga al menos un CV")
    cfg = {
        "supervisor_prompt": f"Rol: {body.supervisor_rol}\n\n{body.supervisor_prompt}",
        "reclutador_rol": body.reclutador_rol,
        "reclutador_prompt": body.reclutador_prompt,
        "entrevistador_rol": body.entrevistador_rol,
        "entrevistador_prompt": body.entrevistador_prompt,
    }
    graph = build_graph(cfg)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    pid = config["configurable"]["thread_id"][:8]
    PROCESOS[pid] = {"graph": graph, "config": config}
    data = {"puesto": body.puesto, "caracteristicas": body.caracteristicas,
            "cvs": [c.model_dump() for c in body.cvs]}
    _run_background(graph, {"messages": [mensaje_inicial(data)]}, config,
                    proc=PROCESOS[pid])
    return {"pid": pid, "estado": "iniciado (pausara en la aprobacion del shortlist)"}

@app.get("/procesos/{pid}")
def estado_proceso(pid: str):
    p = PROCESOS.get(pid)
    if not p:
        raise HTTPException(404, "pid no encontrado")
    if p.get("error"):
        return {"pid": pid, "estado": "error", "error": p["error"],
                "nodo_actual": [], "total_mensajes": 0, "pendiente_aprobacion": False}
    if p.get("running"):
        return {"pid": pid, "estado": "trabajando", "nodo_actual": [],
                "total_mensajes": 0, "pendiente_aprobacion": False}
    snap = _snapshot(p["graph"], p["config"])
    return {"pid": pid, **snap}

@app.post("/procesos/{pid}/respuesta")
async def responder(pid: str, body: Respuesta):
    p = PROCESOS.get(pid)
    if not p:
        raise HTTPException(404, "pid no encontrado")
    if p.get("running") or p.get("error") or not _snapshot(p["graph"], p["config"])["pendiente_aprobacion"]:
        raise HTTPException(409, "No hay aprobacion pendiente")
    p.pop("error", None)
    _run_background(p["graph"], Command(resume=body.respuesta), p["config"], proc=p)
    return {"pid": pid, "estado": "continuando (entrevistador trabajando)"}

app.mount("/panel", StaticFiles(directory="frontend", html=True), name="panel")
