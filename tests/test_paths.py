from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wheeled_uav.paths import GENERATED_XML_FILENAME, PathResolver


class PathResolverTests(unittest.TestCase):
    def test_generated_xml_defaults_to_build_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            resolver = PathResolver(repo_root=repo_root)

            generated_path = resolver.get_generated_xml_path(0)

            self.assertEqual(generated_path, repo_root / "build" / "generated_xml" / GENERATED_XML_FILENAME)
            self.assertTrue(generated_path.parent.exists())

    def test_generated_xml_uses_instance_specific_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = PathResolver(repo_root=Path(temp_dir))

            generated_path = resolver.get_generated_xml_path(3)

            self.assertEqual(generated_path.name, "wheeled_uav.generated.instance_3.xml")

    def test_environment_override_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            override_dir = repo_root / "artifacts"
            resolver = PathResolver(repo_root=repo_root)

            with patch.dict(os.environ, {resolver.generated_xml_dir_env_var: str(override_dir)}, clear=False):
                generated_dir = resolver.get_generated_xml_dir()

            self.assertEqual(generated_dir, override_dir.resolve())


if __name__ == "__main__":
    unittest.main()