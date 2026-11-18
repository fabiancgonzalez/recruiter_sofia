# Reclutador multiagente v2 (LangGraph + FastAPI)

Equipo: Supervisor Recruiter -> Reclutador (screening CVs vs puesto)
-> [pausa humana: aprobar shortlist] -> Entrevistador (preguntas
personalizadas por candidato) -> fin.

Roles y prompts de los 3 agentes editables desde el panel; el grafo se
construye dinamico por proceso con build_graph(cfg). El supervisor enruta
al inicio; al terminar el entrevistador, no se vuelve a consultar al supervisor.

## Instalacion
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn api:app --reload --port 8000

## Despliegue en Railway
Usá la raíz de este repositorio como Root Directory del servicio y Railpack
como builder. `railpack.json` define el arranque de FastAPI con:

    python -m uvicorn api:app --host 0.0.0.0 --port $PORT

Railway proporciona `PORT`. No uses `--reload` en producción ni varios workers:
el estado de los procesos activos se mantiene en memoria.
Si configuraste un Start Command manual en Railway, borralo para usar el del
repositorio o reemplazalo por el comando anterior.

En Variables, configurá `PROVIDER`, `MODEL` y la clave del proveedor elegido
(ver la sección siguiente). Subí estos cambios al repositorio conectado y
volvé a desplegar. Generá un dominio público y abrí `/panel`.

Si una etapa no termina, abrí **Deploy Logs** del servicio y buscá `Proceso`
o el ID que muestra el panel. El servidor registra el inicio, la duración y
el tipo de fallo de cada etapa, sin incluir CVs ni claves. Las solicitudes
al modelo tienen un timeout de 60 segundos y no se reintentan automáticamente.

El disco del contenedor es efímero. Para conservar los checkpoints entre
despliegues, montá un volumen, por ejemplo en `/data`, y configurá
`CHECKPOINT_DB=/data/checkpoints_reclutador.sqlite`. Esto conserva los
checkpoints, pero no recupera los procesos activos en memoria tras un reinicio.

## Deteccion de FastAPI en Vercel
`pyproject.toml` declara `api:app` como entrypoint para evitar el error
`FASTAPI_ENTRYPOINT_NOT_FOUND`. Usá la raíz del repositorio como Root Directory
y el preset FastAPI. Las dependencias siguen en `requirements.txt`.

Esto corrige la detección, pero no adapta el flujo completo a Vercel Functions:
los procesos activos viven en memoria y pueden perderse o no estar disponibles
en otra instancia; el trabajo en segundo plano no tiene garantizada su
continuidad después de responder. SQLite tampoco puede escribirse en el
directorio de la aplicación; `/tmp` es temporal y no compartido.
Para producción con la arquitectura actual, usá Railway. Vercel AI Gateway
puede utilizarse desde Railway sin desplegar la aplicación en Vercel.

## Proveedor del modelo
Configurá uno de estos proveedores en `.env` y reiniciá el servidor al cambiarlo:

Para usar Vercel AI Gateway:

    PROVIDER=vercel_ai_gateway
    VERCEL_AI_GATEWAY_API_KEY=tu_clave_de_vercel_ai_gateway
    MODEL=reflection/beam-free

También se acepta el nombre estándar `AI_GATEWAY_API_KEY`.
`reflection/beam-free` es un modelo del nivel gratuito con soporte para
herramientas y salida estructurada. Otros modelos pueden requerir créditos.

Para usar Groq directamente:

    PROVIDER=groq
    GROQ_API_KEY=tu_clave_de_groq
    MODEL=llama-3.3-70b-versatile

Para usar OpenAI:

    PROVIDER=openai
    OPENAI_API_KEY=tu_clave_de_openai
    MODEL=gpt-5-mini

Definí `MODEL` una sola vez: debe ser compatible con el proveedor elegido.
Por defecto se usa OpenAI. No compartas ni subas las claves al repositorio.
Los CVs y los datos del puesto se envían al proveedor seleccionado para su análisis.
El acceso a Llama 3.3 depende de los permisos y límites de tu cuenta de Groq.
Al iniciar, el servidor muestra el proveedor y el modelo efectivos, sin mostrar
claves. Si el modelo devuelve 404, verificá esos valores y la disponibilidad
del modelo en tu cuenta. Las variables del entorno tienen prioridad sobre `.env`.

## Uso
    http://localhost:8000/panel
    1. editar roles/prompts del supervisor, recruiter y entrevistador
    2. completar puesto y caracteristicas
    3. cargar CVs: pegar texto, subir .txt/.md, o subir PDF (extraccion
       con pypdf via POST /cvs/pdf)
    4. iniciar -> revisar el análisis y seleccionar un candidato en la grilla
    5. opcionalmente abrir un borrador de correo o un evento editable de
       Google Calendar; completar destinatario o fecha y confirmar manualmente
    6. aprobar shortlist o pedir ajustes -> ver preguntas

## API
    POST /procesos            crear proceso (JSON, ver api.py)
    POST /cvs/pdf             multipart files=[pdf...] -> [{nombre, texto}]
    GET  /procesos/{pid}      estado / shortlist / resultado
    POST /procesos/{pid}/respuesta   {"respuesta": "si" | "ajustes..."}

## Notas
- PDFs escaneados (sin capa de texto) dan error 400 con mensaje claro.
- Persistencia en checkpoints_reclutador.sqlite por thread_id.
- Si OpenAI informa que no hay créditos, revisá la facturación de la cuenta
  utilizada por `OPENAI_API_KEY` y reiniciá el proceso desde el panel.
  Los CVs y la configuración permanecen cargados al volver a editar.
- Groq también puede informar errores de autenticación, permisos o límites de
  solicitudes; revisá la cuenta y el modelo configurado antes de reintentar.
- La selección en la grilla es manual. Los borradores incluyen el nombre,
  el puesto y las notas que ingreses, pero no los CVs ni el análisis completo.


npx vercel@latest ai-gateway setup