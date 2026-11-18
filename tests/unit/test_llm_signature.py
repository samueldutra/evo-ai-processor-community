"""The model's arguments must reach the HTTP tool.

google-adk 1.19 filters the call arguments down to the names in the function
signature. The tools were `def http_tool(**kwargs)`, so every argument the model
sent was dropped and the configured defaults went out on the wire.
"""

import asyncio
import inspect
from unittest.mock import MagicMock, patch

import pytest

from src.services.adk.custom_tools import CustomToolBuilder, llm_signature
from src.services.adk.tool_builder import ToolBuilder

BUILDERS = [CustomToolBuilder, ToolBuilder]


def _config(**parameters):
    return {
        "name": "buscar_produto",
        "description": "Searches products",
        "method": "GET",
        "endpoint": "https://api.example.com/search/products",
        "headers": {},
        "parameters": parameters,
        "values": {},
    }


class TestLlmSignature:
    def test_exposes_only_placeholder_query_params(self):
        sig = llm_signature(
            {"query_params": {"q": "{q}", "per_page": "3", "fields": "id,name"}}
        )

        assert list(sig.parameters) == ["q"]

    def test_none_when_nothing_is_dynamic(self):
        assert llm_signature({"query_params": {"per_page": "3"}}) is None
        assert llm_signature({}) is None
        assert llm_signature(None) is None

    def test_path_params_are_exposed_and_required(self):
        sig = llm_signature({"path_params": {"id": "x"}})

        assert sig.parameters["id"].default is inspect.Parameter.empty

    def test_names_that_are_not_identifiers_stay_static(self):
        sig = llm_signature(
            {"query_params": {"filter[Cor]": "{filter[Cor]}", "q": "{q}"}}
        )

        assert list(sig.parameters) == ["q"]


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
class TestModelArgumentsReachTheWire:
    def _run(self, builder, config, args):
        built = builder()._create_http_tool(config)
        target = f"{builder.__module__}.requests.request"
        with patch(target) as request:
            request.return_value = MagicMock(status_code=200, json=lambda: {"ok": 1})
            asyncio.run(built.run_async(args=args, tool_context=MagicMock()))
        return request.call_args.kwargs

    def test_the_model_search_text_replaces_the_placeholder(self, builder):
        sent = self._run(
            builder,
            _config(query_params={"q": "{q}", "per_page": "3"}),
            {"q": "wahl magic clip"},
        )

        assert sent["params"]["q"] == "wahl magic clip"
        assert sent["params"]["per_page"] == "3"

    def test_a_static_param_cannot_be_overridden_by_the_model(self, builder):
        sent = self._run(
            builder,
            _config(query_params={"q": "{q}", "fields": "id,name"}),
            {"q": "x", "fields": "variants"},
        )

        assert sent["params"]["fields"] == "id,name"

    def test_tools_without_dynamic_params_still_work(self, builder):
        sent = self._run(builder, _config(query_params={"per_page": "3"}), {})

        assert sent["params"] == {"per_page": "3"}
