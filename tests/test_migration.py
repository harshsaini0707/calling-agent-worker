"""Guard tests for the LiveKit Cloud migration (branch: lk-cloud-migration).

These cover the wiring the migration touched without requiring a LiveKit server,
network, or provider API keys:
  - module imports with no side effects (no log-dir creation, no network)
  - AgentServer entrypoint replaces the old WorkerOptions pattern
  - the dispatch agent_name stays "outbound-caller"
  - prewarm is registered as the server setup_fnc
  - the optional noise_cancellation import is guarded
  - pure config helpers still behave
"""

from livekit.agents import AgentServer

import agent


def test_module_imports_cleanly():
    # If import had side effects (e.g. creating /app/logs) this would already
    # have failed during collection. Assert the key symbols exist.
    assert hasattr(agent, "entrypoint")
    assert hasattr(agent, "prewarm")
    assert hasattr(agent, "server")


def test_uses_agent_server():
    assert isinstance(agent.server, AgentServer)


def test_prewarm_is_registered_setup_fnc():
    assert agent.server.setup_fnc is agent.prewarm


def test_agent_name_is_outbound_caller():
    # agent_name must stay "outbound-caller" so the backend/make_call dispatch
    # finds this worker. It is set on the @server.rtc_session decorator; assert
    # the literal is still present in the source as a regression guard.
    import inspect

    src = inspect.getsource(agent)
    assert 'agent_name="outbound-caller"' in src
    assert "zariya-interviewer" not in src


def test_noise_cancellation_import_is_guarded():
    # Whether or not the plugin is installed, the symbol must exist so the
    # entrypoint's `if noise_cancellation is not None` guard works.
    assert hasattr(agent, "noise_cancellation")


def test_bulbul_model_selection():
    assert agent.get_bulbul_model("anushka") == "bulbul:v2"
    assert agent.get_bulbul_model("simran") == "bulbul:v3-beta"


def test_coerce_tts_pace_falls_back_on_garbage():
    assert agent._coerce_tts_pace("0.9") == 0.9
    assert agent._coerce_tts_pace("not-a-number") == 0.95


def test_normalize_ai_config_defaults():
    cfg = agent._normalize_ai_config({})
    assert cfg["version"] == 2
    assert "provider" in cfg["stt"]
    assert "provider" in cfg["llm"]
    assert "provider" in cfg["tts"]
