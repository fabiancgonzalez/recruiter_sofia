"""
API del sistema reclutador multiagente (v2).

POST /procesos              {"puesto", "caracteristicas", "cvs": [{nombre, texto}],
                             "supervisor_rol", "supervisor_prompt",
                             "reclutador_rol", "reclutador_prompt",
                             "entrevistador_rol", "entrevistador_prompt"} -> pid
POST /cvs/pdf               multipart: files=[.pdf...] -> [{"nombre", "texto"}]
GET  /procesos/{pid}        estado + shortlist/preguntas cuando esta en pausa
POST /procesos/{pid}/respuesta  {"respuesta": "si" | "ajustes..."}
POST /procesos/{pid}/resumen    {"seleccion": [0, ...], "notas": "..."} -> .docx
"""
import os, io, asyncio, uuid, logging, re
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from time import monotonic
from zipfile import ZIP_DEFLATED, ZipFile
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, StrictInt
from openai import APITimeoutError, AuthenticationError, NotFoundError, PermissionDeniedError, RateLimitError

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
log = logging.getLogger("uvicorn.error")


def _run_background(graph, *args, proc, pid: str, fase: str):
    """Ejecuta graph.invoke en un hilo sin bloquear la respuesta HTTP."""
    proc["running"] = True
    proc.pop("error", None)
    inicio = monotonic()
    log.info("Proceso %s: inicio de %s", pid, fase)

    def _on_done(fut):
        exc = fut.exception()
        duracion = monotonic() - inicio
        try:
            if exc:
                log.error("Proceso %s: fallo en %s después de %.1f s (%s)",
                          pid, fase, duracion, type(exc).__name__)
                proveedor = {
                    "groq": "Groq",
                    "vercel_ai_gateway": "Vercel AI Gateway",
                }.get(PROVIDER, "OpenAI")
                if isinstance(exc, APITimeoutError):
                    proc["error"] = f"{proveedor} tardó demasiado en responder. Volvé a intentar."
                elif isinstance(exc, RateLimitError):
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
            else:
                log.info("Proceso %s: terminó %s en %.1f s", pid, fase, duracion)
        finally:
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

    supervisor_rol: str = "Supervisor de Reclutamiento S3"
    supervisor_prompt: str = (
        "Sos el Supervisor de Reclutamiento de Strategic Staffing Solutions (S3). "
        "Coordinas procesos de seleccion end-to-end para clientes globales y startups de EE.UU. "
        "Tu equipo: Sofia Gonzalez, Senior Tech Recruiter especializada en Data & AI y OT/IT, "
        "y un Entrevistador Tecnico. Tu responsabilidad es la calidad, la trazabilidad y los tiempos. "
        "Reglas: "
        "1) Delegas el screening en Sofia. Confias en su criterio, pero revisas que el ranking "
        "este anclado a evidencia del CV y sea consistente con las caracteristicas del puesto. "
        "2) No avanzas sin aprobacion humana del shortlist. Si el humano pide ajustes, los "
        "traduces en instrucciones concretas para Sofia y volves a pedir el shortlist. "
        "3) Con el shortlist aprobado, delegas en el Entrevistador Tecnico la generacion de "
        "preguntas por candidato. "
        "4) Controlas que ninguna recomendacion asuma datos ausentes: si detectas una suposicion, "
        "la marcas y la corriges antes de escalar. "
        "5) Cuando las preguntas esten listas y validadas, respondes FINALIZAR. "
        "Comunicas en formato claro: estado del paso, decision tomada, siguiente paso."
    )

    reclutador_rol: str = "Sofia Gonzalez - Senior Tech Recruiter de S3"
    reclutador_prompt: str = (
        "Sos Sofia Gonzalez, Senior Tech Recruiter de Strategic Staffing Solutions (S3). "
        "Tenés 5+ años de reclutamiento tecnico full-cycle para staffing y consultoria nearshore, "
        "cubriendo LATAM (Argentina, Uruguay, Colombia, Peru, Costa Rica, Republica Dominicana), "
        "Filipinas y EE.UU., para clientes enterprise multinacionales y startups de EE.UU. "
        "Tu especialidad es talento Data & AI: Data Engineers, Solutions Architects, ML Engineers, "
        "Technical Product Managers, Computer Vision y Full Stack Developers que construyen "
        "agentes de IA con Semantic Kernel, LangGraph y Azure OpenAI. Tambien reclutas OT/IT "
        "industrial (Energia, Oil & Gas, Manufactura, Utilities) y perfiles Cloud, DevOps, SRE, "
        "Cybersecurity y QA. "
        "Herramientas y metodo: sourcing con LinkedIn Recruiter, HireEZ y Boolean search; "
        "screening tecnico asistido por IA para evaluar alcance real y seniority antes de la "
        "entrevista con el cliente; seguimiento de metricas (time-to-hire, interview failure rate). "
        "Tu background en Psicologia (UNC) define tu estilo: escuchas activamente, lees entre "
        "lineas y cuidas la candidate experience. "
        "Tarea: recibis el puesto con sus caracteristicas (stack, seniority, modalidad, rango "
        "salarial, idiomas, industria) y una lista de CVs. Para cada candidato evalua y documenta: "
        "(a) match de habilidades tecnicas contra requisitos must-have y nice-to-have, "
        "distinguiendo stack, profundidad de uso y contexto (producto, escala, industria); "
        "(b) experiencia relevante: proyectos, responsabilidades, impacto medible si figura; "
        "(c) seniority real vs. seniority declarado, segun alcance tecnico y liderazgo; "
        "(d) expectativa salarial y modalidad, solo si figuran en el CV; "
        "(e) senales objetivas de riesgo (gaps, rotacion, inconsistencias), sin especular. "
        "Entrega: 1) ranking de mejor a peor con puntaje 1-10 y justificacion corta citando "
        "evidencia concreta del CV; 2) shortlist de los 3 mejores; 3) para cada uno de esos 3, "
        "3 a 5 puntos concretos a profundizar en la entrevista tecnica, alineados a su stack "
        "y a las necesidades del puesto. "
        "Reglas de estilo: tono profesional, lenguaje inclusivo, sin adjetivos grandilocuentes. "
        "Si un dato no esta en el CV, escribi 'no informado' y no lo asumas. "
        "Si el CV muestra experiencia en OT (SCADA, IoT industrial, redes electricas, "
        "integracion IT/OT) o en agentes de IA (Semantic Kernel, LangGraph, Azure OpenAI), "
        "destacalo explicitamente por su alta demanda."
    )

    entrevistador_rol: str = "Entrevistador Tecnico S3"
    entrevistador_prompt: str = (
        "Sos el Entrevistador Tecnico de S3. Trabajas sobre el shortlist aprobado por el humano "
        "y los CVs originales. Tu objetivo es validar con evidencia competencias tecnicas y "
        "conductuales, sin repetir lo que ya esta claro en el CV. "
        "Para cada candidato seleccionado genera: "
        "1) 5 preguntas tecnicas personalizadas, ancladas a su stack y a los proyectos que "
        "declara, con foco en profundidad: decisiones de diseno, trade-offs, debugging, "
        "performance, testing y, si aplica, integracion de IA (Semantic Kernel, LangGraph, "
        "Azure OpenAI) o entornos OT/IT (SCADA, IoT industrial, ciberseguridad industrial). "
        "2) 3 preguntas situacionales/soft skills orientadas a colaboracion, comunicacion "
        "con stakeholders globales, manejo de ambiguiedad y candidate experience. "
        "3) 2 red flags a verificar, formuladas como preguntas o verificaciones concretas, "
        "derivadas de vacios o inconsistencias del CV (no de suposiciones). "
        "Organiza la salida por candidato, con encabezado claro. Para cada pregunta indica: "
        "competencia evaluada y que respuesta seria considerada solida. "
        "Mantene tono profesional y lenguaje inclusivo. Evita preguntas discriminatorias, "
        "ilegales o que revelen informacion sensible no relacionada con el rol."
    )

class Respuesta(BaseModel):
    respuesta: str

class ResumenWord(BaseModel):
    seleccion: list[StrictInt]
    notas: str = ""

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _resumen_docx(puesto: str, nombres: list[str], notas: str, resultado: str) -> bytes:
    def parrafos(texto: str, puntos: int = 0) -> str:
        texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]", "", texto)
        formato = f'<w:rPr><w:b/><w:sz w:val="{puntos * 2}"/></w:rPr>' if puntos else ""
        return "".join(f'<w:p><w:r>{formato}<w:t xml:space="preserve">{escape(linea)}</w:t></w:r></w:p>'
                       for linea in texto.split("\n"))

    contenido = parrafos(f"Resumen de selección: {puesto}", 20)
    contenido += parrafos("Candidatos seleccionados", 14)
    contenido += "".join(parrafos(nombre, 16) for nombre in nombres) if nombres else parrafos("Sin selección.")
    if notas:
        contenido += parrafos("Notas", 14) + parrafos(notas)
    contenido += parrafos("Resultado generado", 14) + parrafos(resultado)
    # Un .docx mínimo consta del manifiesto, la relación raíz y el documento.
    archivo = io.BytesIO()
    with ZipFile(archivo, "w", ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml",
                      '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                      '<Override PartName="/word/document.xml" '
                      f'ContentType="{DOCX_MIME}.main+xml"/></Types>')
        docx.writestr("_rels/.rels",
                      '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                      '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                      'Target="word/document.xml"/></Relationships>')
        docx.writestr("word/document.xml",
                      '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                      f'<w:body>{contenido}</w:body></w:document>')
    return archivo.getvalue()

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
    if isinstance(resultado, list):
        resultado = "\n".join(
            bloque if isinstance(bloque, str) else bloque.get("text", "")
            for bloque in resultado
            if isinstance(bloque, str) or
            (bloque.get("type") == "text" and isinstance(bloque.get("text"), str))
        )
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

@app.api_route("/panel", methods=["GET", "HEAD"], response_class=HTMLResponse, include_in_schema=False)
@app.api_route("/panel/", methods=["GET", "HEAD"], response_class=HTMLResponse, include_in_schema=False)
@app.api_route("/panel/index.html", methods=["GET", "HEAD"], response_class=HTMLResponse, include_in_schema=False)
def panel() -> str:
    plantilla = Path(__file__).with_name("frontend").joinpath("index.html").read_text(encoding="utf-8")
    return re.sub(
        r"\{\{ (\w+) \}\}",
        lambda marca: escape(NuevoProceso.model_fields[marca[1]].default),
        plantilla,
    )

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
    PROCESOS[pid] = {"graph": graph, "config": config, "puesto": body.puesto,
                    "nombres": [cv.nombre for cv in body.cvs]}
    data = {"puesto": body.puesto, "caracteristicas": body.caracteristicas,
            "cvs": [c.model_dump() for c in body.cvs]}
    _run_background(graph, {"messages": [mensaje_inicial(data)]}, config,
                    proc=PROCESOS[pid], pid=pid, fase="screening")
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

@app.post("/procesos/{pid}/resumen", response_class=Response,
          responses={200: {"content": {DOCX_MIME: {}}}})
def descargar_resumen(pid: str, body: ResumenWord):
    proceso = estado_proceso(pid)
    if proceso["estado"] != "finalizado":
        raise HTTPException(409, "El resumen está disponible cuando finaliza el proceso.")
    p = PROCESOS[pid]
    nombres = p["nombres"]
    indices = body.seleccion
    if len(set(indices)) != len(indices) or any(not 0 <= i < len(nombres) for i in indices):
        raise HTTPException(400, "La selección contiene candidatos repetidos o que no pertenecen al proceso.")
    contenido = _resumen_docx(p["puesto"], [nombres[i] for i in indices], body.notas, proceso["resultado"])
    return Response(contenido, media_type=DOCX_MIME, headers={"Cache-Control": "no-store"})

@app.post("/procesos/{pid}/respuesta")
async def responder(pid: str, body: Respuesta):
    p = PROCESOS.get(pid)
    if not p:
        raise HTTPException(404, "pid no encontrado")
    if p.get("running") or p.get("error") or not _snapshot(p["graph"], p["config"])["pendiente_aprobacion"]:
        raise HTTPException(409, "No hay aprobacion pendiente")
    _run_background(p["graph"], Command(resume=body.respuesta), p["config"],
                    proc=p, pid=pid, fase="continuación")
    return {"pid": pid, "estado": "continuando (entrevistador trabajando)"}
