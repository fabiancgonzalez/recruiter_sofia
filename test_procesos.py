import asyncio
import json
import os
import threading
import unittest
from html import escape
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree
from zipfile import ZipFile

import httpx
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from openai import APITimeoutError, NotFoundError, PermissionDeniedError, RateLimitError

import api
import graph_factory


class PanelTests(unittest.TestCase):
    def test_panel_prefills_defaults_for_the_three_agents(self):
        response = TestClient(api.app).get("/panel")
        self.assertEqual(response.status_code, 200)
        defaults = api.NuevoProceso(puesto="", caracteristicas="", cvs=[])
        for prefijo, agente in (("sup", "supervisor"), ("rec", "reclutador"), ("ent", "entrevistador")):
            with self.subTest(agente=agente):
                rol = escape(getattr(defaults, f"{agente}_rol"))
                prompt = escape(getattr(defaults, f"{agente}_prompt"))
                self.assertIn(f'id="{prefijo}-rol" value="{rol}"', response.text)
                self.assertRegex(response.text, rf'<textarea id="{prefijo}-prompt"[^>]*>')
                self.assertIn(f">{prompt}</textarea>", response.text)

    def test_panel_escapes_default_text_as_html(self):
        fields = api.NuevoProceso.model_fields
        rol = 'Rol "A" & "B"'
        prompt = 'Escribi <b>HTML</b> & texto.'
        with patch.object(fields["supervisor_rol"], "default", rol), \
             patch.object(fields["supervisor_prompt"], "default", prompt):
            response = TestClient(api.app).get("/panel/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="sup-rol" value="Rol &quot;A&quot; &amp; &quot;B&quot;"', response.text)
        self.assertIn('>Escribi &lt;b&gt;HTML&lt;/b&gt; &amp; texto.</textarea>', response.text)


class ResumenTests(unittest.TestCase):
    def setUp(self):
        self.estado = SimpleNamespace(
            values={"messages": [AIMessage(content=[{"type": "text", "text": "¿Cómo probás el código?"}])]},
            next=(),
        )
        self.proceso = {
            "graph": SimpleNamespace(get_state=lambda config: self.estado),
            "config": {},
            "puesto": "Diseño & Datos",
            "nombres": ["Ana", "Bruno", "Cecilia", "Daniel", "Eva"],
        }
        parche = patch.dict(api.PROCESOS, {"docx-test": self.proceso})
        parche.start()
        self.addCleanup(parche.stop)
        self.client = TestClient(api.app)
        self.url = "/procesos/docx-test/resumen"

    def test_descarga_word_con_mas_de_tres_seleccionados_y_resultado(self):
        r = self.client.post(self.url, json={"seleccion": [0, 2, 3, 4], "notas": "Primera línea\n<Segunda>"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], api.DOCX_MIME)
        with ZipFile(BytesIO(r.content)) as docx:
            self.assertIsNone(docx.testzip())
            documento = ElementTree.fromstring(docx.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        parrafos = {p.findtext("w:r/w:t", namespaces=ns): p for p in documento.findall(".//w:p", ns)}
        self.assertNotIn("Bruno", parrafos)
        for texto in ("Resumen de selección: Diseño & Datos", "Primera línea", "<Segunda>", "¿Cómo probás el código?"):
            self.assertIn(texto, parrafos)
        for nombre in ("Ana", "Cecilia", "Daniel", "Eva"):
            self.assertIsNotNone(parrafos[nombre].find("w:r/w:rPr/w:b", ns))
            tamano = parrafos[nombre].find("w:r/w:rPr/w:sz", ns)
            self.assertGreater(int(tamano.get(f"{{{ns['w']}}}val")), 22)

    def test_rechaza_indices_invalidos_o_duplicados_y_permite_seleccion_vacia(self):
        for seleccion in ([-1], [5], [0, 0]):
            with self.subTest(seleccion=seleccion):
                self.assertEqual(self.client.post(self.url, json={"seleccion": seleccion}).status_code, 400)
        self.assertEqual(self.client.post(self.url, json={"seleccion": []}).status_code, 200)

    def test_no_descarga_resumen_antes_de_finalizar(self):
        self.estado.next = ("aprobacion",)
        self.assertEqual(self.client.post(self.url, json={"seleccion": [0]}).status_code, 409)


class BackgroundTests(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_free_tier_error_suggests_free_model(self):
        response = httpx.Response(
            403, request=httpx.Request("POST", "https://ai-gateway.vercel.sh/v1/chat/completions"))
        error = PermissionDeniedError(
            "Free tier users do not have access to this model", response=response, body=None)
        proc = {"graph": SimpleNamespace(invoke=lambda *args: (_ for _ in ()).throw(error)),
                "config": {}}
        api.PROCESOS["free-tier"] = proc
        self.addCleanup(api.PROCESOS.pop, "free-tier", None)
        with patch.object(api.log, "error"), \
             patch.object(api, "PROVIDER", "vercel_ai_gateway"), \
             patch.object(api, "model", SimpleNamespace(model_name="deepseek/deepseek-v4.1-flash-1017")):
            api._run_background(proc["graph"], proc=proc, pid="free-tier", fase="screening")
            for _ in range(100):
                if not proc["running"]:
                    break
                await asyncio.sleep(0.01)
        result = api.estado_proceso("free-tier")
        self.assertIn("Vercel AI Gateway", result["error"])
        self.assertIn("reflection/beam-free", result["error"])

    async def test_process_reports_running_then_quota_error(self):
        release = threading.Event()
        response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"))
        error = RateLimitError("You have no credits remaining", response=response, body=None)

        def invoke(*args):
            release.wait(timeout=5)
            raise error

        proc = {"graph": SimpleNamespace(invoke=invoke), "config": {}}
        api.PROCESOS["test"] = proc
        self.addCleanup(api.PROCESOS.pop, "test", None)
        with patch.object(api.log, "error"), patch.object(api, "PROVIDER", "groq"):
            api._run_background(proc["graph"], proc=proc, pid="test", fase="screening")
            self.assertEqual(api.estado_proceso("test")["estado"], "trabajando")
            release.set()
            for _ in range(100):
                if not proc["running"]:
                    break
                await asyncio.sleep(0.01)
        result = api.estado_proceso("test")
        self.assertEqual(result["estado"], "error")
        self.assertIn("Groq no tiene créditos", result["error"])
        self.assertFalse(result["pendiente_aprobacion"])

    async def test_timeout_during_interview_stops_running_and_reports_error(self):
        error = APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))

        def invoke(*args):
            raise error

        proc = {"graph": SimpleNamespace(invoke=invoke), "config": {}}
        api.PROCESOS["interview"] = proc
        self.addCleanup(api.PROCESOS.pop, "interview", None)
        with self.assertLogs(api.log, level="INFO") as logs, \
             patch.object(api, "PROVIDER", "groq"):
            api._run_background(proc["graph"], proc=proc, pid="interview", fase="continuación")
            for _ in range(100):
                if not proc["running"]:
                    break
                await asyncio.sleep(0.01)
        result = api.estado_proceso("interview")
        self.assertEqual(result["estado"], "error")
        self.assertIn("Groq tardó demasiado", result["error"])
        self.assertIn("Proceso interview", logs.output[-1])
        self.assertIn("APITimeoutError", logs.output[-1])

    def test_empty_checkpoint_is_not_a_success(self):
        for messages in ([], [HumanMessage(content="CV")]):
            graph = SimpleNamespace(get_state=lambda config: SimpleNamespace(
                values={"messages": messages}, next=()))
            snapshot = api._snapshot(graph, {})
            self.assertEqual(snapshot["estado"], "error")
            self.assertIsNone(snapshot["resultado"])

    async def test_model_not_found_reports_provider_and_model(self):
        for host, provider in (("api.groq.com", "Groq"), ("api.openai.com", "OpenAI")):
            response = httpx.Response(404, request=httpx.Request("POST", f"https://{host}/chat/completions"))
            error = NotFoundError("model not found", response=response, body=None)

            def invoke(*args):
                raise error

            proc = {"graph": SimpleNamespace(invoke=invoke), "config": {}}
            api.PROCESOS["missing-model"] = proc
            self.addCleanup(api.PROCESOS.pop, "missing-model", None)
            with patch.object(api.log, "error"), patch.object(api, "PROVIDER", "groq"), \
                 patch.object(api, "model", SimpleNamespace(model_name="llama-3.3-70b-versatile")):
                api._run_background(proc["graph"], proc=proc, pid="missing-model", fase="screening")
                for _ in range(100):
                    if not proc["running"]:
                        break
                    await asyncio.sleep(0.01)
                result = api.estado_proceso("missing-model")
            self.assertEqual(result["estado"], "error")
            self.assertIn(provider, result["error"])
            self.assertIn("llama-3.3-70b-versatile", result["error"])
            self.assertIn("no tiene acceso", result["error"])


class GraphTests(unittest.TestCase):
    def test_supervisor_route_survives_state_and_resume(self):
        class Router:
            def __init__(self):
                self.decisions = iter(["reclutador"])

            def invoke(self, messages):
                return graph_factory.RouteDecision(next=next(self.decisions))

        class Agent:
            def __init__(self, name):
                self.name = name

            def invoke(self, state):
                return {"messages": [AIMessage(content=f"Resultado {self.name}", name=self.name)]}

        model = SimpleNamespace(with_structured_output=lambda schema, **kwargs: Router())
        cfg = {"supervisor_prompt": "", "reclutador_rol": "", "reclutador_prompt": "",
               "entrevistador_rol": "", "entrevistador_prompt": ""}
        with patch.object(graph_factory, "model", model), \
             patch.object(graph_factory, "CHECKPOINT_DB", ":memory:"), \
             patch.object(graph_factory, "create_react_agent",
                          side_effect=lambda *args, name, **kwargs: Agent(name)):
            graph = graph_factory.build_graph(cfg)
            config = {"configurable": {"thread_id": "test"}}
            graph.invoke({"messages": [HumanMessage(content="CV")]}, config)
            self.assertTrue(api._snapshot(graph, config)["pendiente_aprobacion"])
            graph.invoke(Command(resume="si"), config)
            snapshot = api._snapshot(graph, config)
            self.assertEqual(snapshot["estado"], "finalizado")
            self.assertEqual(snapshot["resultado"], "Resultado entrevistador")

    def test_groq_graph_uses_compatible_requests(self):
        requests = []
        routes = iter(["reclutador"])

        def respond(request):
            payload = json.loads(request.content)
            requests.append((request, payload))
            if "tools" in payload:
                decision = next(routes)
                message = {
                    "role": "assistant", "content": None,
                    "tool_calls": [{"id": "call_1", "type": "function",
                                    "function": {"name": "RouteDecision",
                                                 "arguments": json.dumps({"next": decision})}}],
                }
                reason = "tool_calls"
            else:
                message = {"role": "assistant", "content": "Respuesta de prueba"}
                reason = "stop"
            return httpx.Response(200, json={
                "id": "test", "object": "chat.completion", "created": 0,
                "model": "llama-3.3-70b-versatile",
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
            })

        client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(client.close)
        model = ChatOpenAI(model="llama-3.3-70b-versatile", api_key="test-key",
                           base_url="https://api.groq.com/openai/v1", http_client=client)
        cfg = {"supervisor_prompt": "", "reclutador_rol": "", "reclutador_prompt": "",
               "entrevistador_rol": "", "entrevistador_prompt": ""}
        with patch.object(graph_factory, "PROVIDER", "groq"), \
             patch.object(graph_factory, "model", model), \
             patch.object(graph_factory, "CHECKPOINT_DB", ":memory:"):
            graph = graph_factory.build_graph(cfg)
            config = {"configurable": {"thread_id": "test-groq"}}
            graph.invoke({"messages": [HumanMessage(content="CV ficticio")]}, config)
            self.assertTrue(api._snapshot(graph, config)["pendiente_aprobacion"])
            graph.invoke(Command(resume="si"), config)
            self.assertEqual(api._snapshot(graph, config)["estado"], "finalizado")

        self.assertEqual(len(requests), 3)
        for request, payload in requests:
            self.assertEqual(str(request.url), "https://api.groq.com/openai/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            self.assertEqual(payload["model"], "llama-3.3-70b-versatile")
            self.assertTrue(all("name" not in message for message in payload["messages"]))
            self.assertNotIn("response_format", payload)

    def test_provider_configuration(self):
        with patch.object(graph_factory, "PROVIDER", "groq"), \
             patch.dict(os.environ, {"GROQ_API_KEY": "", "MODEL": "llama-3.3-70b-versatile"}):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                graph_factory.build_model()
        with patch.object(graph_factory, "PROVIDER", "vercel_ai_gateway"), \
             patch.dict(os.environ, {
                 "VERCEL_AI_GATEWAY_API_KEY": "test-key",
                 "AI_GATEWAY_API_KEY": "",
                 "MODEL": "reflection/beam-free",
             }):
            model = graph_factory.build_model()
            self.assertEqual(model.model_name, "reflection/beam-free")
            self.assertEqual(model.openai_api_base, "https://ai-gateway.vercel.sh/v1")
            self.assertEqual(model.openai_api_key.get_secret_value(), "test-key")
        with patch.object(graph_factory, "PROVIDER", "desconocido"):
            with self.assertRaisesRegex(ValueError, "PROVIDER no admitido"):
                graph_factory.build_model()
        with patch.object(graph_factory, "PROVIDER", "openai"), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "MODEL": "gpt-5-mini"}):
            self.assertEqual(graph_factory.build_model().model_name, "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
