from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from backend.graphrag import tracing


class TracingManagerTests(unittest.TestCase):
    def tearDown(self) -> None:
        tracing.TracingManager._instance = None

    def test_del_flushes_and_shuts_down_langfuse(self) -> None:
        langfuse_client = Mock()
        langfuse_client.flush = Mock()
        langfuse_client.shutdown = Mock()

        with patch.object(tracing, "Langfuse", return_value=langfuse_client), patch.dict(
            tracing.os.environ,
            {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"},
            clear=False,
        ):
            tracing.TracingManager._instance = None
            manager = tracing.TracingManager()

        manager.__del__()

        langfuse_client.flush.assert_called_once()
        langfuse_client.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
