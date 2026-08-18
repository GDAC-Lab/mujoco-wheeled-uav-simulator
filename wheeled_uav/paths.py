from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
	"DEFAULT_GENERATED_XML_DIR",
	"DEFAULT_PATH_RESOLVER",
	"GENERATED_XML_FILENAME",
	"GENERATED_XML_PATH",
	"PARAMS_PATH",
	"PathResolver",
	"REPO_ROOT",
	"ROTOR_NAMES",
	"SURFACE_GEOM_NAME",
	"SURFACE_HFIELD_NAME",
	"WALL_GEOM_NAME",
	"XML_TEMPLATE_PATH",
	"get_generated_xml_dir",
	"get_generated_xml_path",
	"get_params_path",
	"get_xml_template_path",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = REPO_ROOT / "vehicle_params.json"
XML_TEMPLATE_PATH = REPO_ROOT / "wheeled_uav.template.xml"
DEFAULT_GENERATED_XML_DIR = REPO_ROOT / "build" / "generated_xml"
GENERATED_XML_FILENAME = "wheeled_uav.generated.xml"
GENERATED_XML_PATH = DEFAULT_GENERATED_XML_DIR / GENERATED_XML_FILENAME
ROTOR_NAMES = ("fr", "fl", "br", "bl")
SURFACE_GEOM_NAME = "surface_geom"
SURFACE_HFIELD_NAME = "generated_surface_hfield"
# Must match the geom name in wheeled_uav.template.xml; the contact report's
# "wall" group and the MATLAB review scripts key off it.
WALL_GEOM_NAME = "wall_geom"

_PARAMS_PATH_ENV_VAR = "WHEELED_UAV_PARAMS_PATH"
_XML_TEMPLATE_PATH_ENV_VAR = "WHEELED_UAV_XML_TEMPLATE_PATH"
_GENERATED_XML_DIR_ENV_VAR = "WHEELED_UAV_GENERATED_XML_DIR"


@dataclass(frozen=True)
class PathResolver:
	repo_root: Path = REPO_ROOT
	params_env_var: str = _PARAMS_PATH_ENV_VAR
	xml_template_env_var: str = _XML_TEMPLATE_PATH_ENV_VAR
	generated_xml_dir_env_var: str = _GENERATED_XML_DIR_ENV_VAR

	@property
	def default_params_path(self) -> Path:
		return self.repo_root / PARAMS_PATH.name

	@property
	def default_xml_template_path(self) -> Path:
		return self.repo_root / XML_TEMPLATE_PATH.name

	@property
	def default_generated_xml_dir(self) -> Path:
		return self.repo_root / DEFAULT_GENERATED_XML_DIR.relative_to(REPO_ROOT)

	def _resolve_path(self, candidate: str | Path | None, env_var_name: str, default_path: Path) -> Path:
		if candidate is not None:
			return Path(candidate).expanduser().resolve()

		env_value = os.environ.get(env_var_name)
		if env_value:
			return Path(env_value).expanduser().resolve()

		return default_path

	def get_params_path(self, params_path: str | Path | None = None) -> Path:
		return self._resolve_path(params_path, self.params_env_var, self.default_params_path)

	def get_xml_template_path(self, template_path: str | Path | None = None) -> Path:
		return self._resolve_path(template_path, self.xml_template_env_var, self.default_xml_template_path)

	def get_generated_xml_dir(self, output_directory: str | Path | None = None, *, create: bool = True) -> Path:
		resolved_directory = self._resolve_path(output_directory, self.generated_xml_dir_env_var, self.default_generated_xml_dir)
		if create:
			resolved_directory.mkdir(parents=True, exist_ok=True)
		return resolved_directory

	def get_generated_xml_path(self, instance_id: int, output_directory: str | Path | None = None) -> Path:
		resolved_directory = self.get_generated_xml_dir(output_directory, create=True)
		if instance_id == 0:
			return resolved_directory / GENERATED_XML_FILENAME
		return resolved_directory / f"wheeled_uav.generated.instance_{instance_id}.xml"


DEFAULT_PATH_RESOLVER = PathResolver()


def get_params_path(params_path: str | Path | None = None) -> Path:
	return DEFAULT_PATH_RESOLVER.get_params_path(params_path)


def get_xml_template_path(template_path: str | Path | None = None) -> Path:
	return DEFAULT_PATH_RESOLVER.get_xml_template_path(template_path)


def get_generated_xml_dir(output_directory: str | Path | None = None, *, create: bool = True) -> Path:
	return DEFAULT_PATH_RESOLVER.get_generated_xml_dir(output_directory, create=create)


def get_generated_xml_path(instance_id: int, output_directory: str | Path | None = None) -> Path:
	return DEFAULT_PATH_RESOLVER.get_generated_xml_path(instance_id, output_directory)
