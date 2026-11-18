# Reclutador multiagente v2 (LangGraph + FastAPI)

Equipo: Supervisor Recruiter -> Reclutador (screening CVs vs puesto)
-> [pausa humana: aprobar shortlist] -> Entrevistador (preguntas
personalizadas por candidato) -> fin.

Roles y prompts de los 3 agentes editables desde el panel; el grafo se
construye dinamico por proceso con build_graph(cfg).

## Instalacion
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn api:app --reload --port 8000

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