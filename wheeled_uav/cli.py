"""Command-line entry point.

Subcommands:
  simulate          Run the MuJoCo simulator (viewer or headless).
  check-model       Render the XML, compile it, print a JSON summary, exit.
  hover-controller  Run the Python reference hover controller over UDP.

Heavy imports (MuJoCo) are deferred into the subcommand handlers so that
``hover-controller`` also works on controller-only hosts without MuJoCo.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version

from .paths import DEFAULT_GENERATED_XML_DIR, REPO_ROOT
from .protocol import PORT_RECV, PORT_SEND

__all__ = ["build_cli_parser", "main", "resolve_params_file"]


def resolve_params_file(params_file: str | None, preset: str | None) -> str | None:
    """--params-file wins; otherwise --preset NAME maps to configs/vehicle_params.NAME.json."""
    if params_file is not None:
        return params_file
    if preset is None:
        return None
    preset_path = REPO_ROOT / "configs" / f"vehicle_params.{preset}.json"
    if not preset_path.is_file():
        # List the names the user can actually type after --preset, not filenames.
        available = sorted(
            candidate.name.removeprefix("vehicle_params.").removesuffix(".json")
            for candidate in (REPO_ROOT / "configs").glob("vehicle_params.*.json")
        )
        raise SystemExit(
            f"error: preset parameter file not found: {preset_path}\n"
            f"available presets: {', '.join(available) if available else '(none)'}"
        )
    return str(preset_path)


def _package_version() -> str:
    try:
        return version("mujoco-wheeled-uav-simulator")
    except PackageNotFoundError:  # editable/source checkout without install metadata
        return "unknown"


def _add_model_arguments(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--instance-id", type=int, default=0, help="Simulation instance id used to derive default ports and XML output")
    subparser.add_argument("--num-uavs", type=int, default=1, help="Number of UAVs to place in a single MuJoCo world")
    subparser.add_argument("--spawn-radius", type=float, default=1.5, help="Radius used to place multiple UAVs around the origin")
    subparser.add_argument("--params-file", default=None, help="Path to a vehicle_params.json file to load instead of the repository default")
    subparser.add_argument("--preset", default=None, metavar="NAME", help="Bundled parameter preset: NAME resolves to configs/vehicle_params.NAME.json (e.g. wall_demo). --params-file wins if both are given")
    subparser.add_argument("--xml-template-file", default=None, help="Path to a MuJoCo XML template file to use instead of wheeled_uav.template.xml")
    subparser.add_argument("--generated-xml-dir", default=None, help=f"Directory to write generated XML files into instead of {DEFAULT_GENERATED_XML_DIR}")
    subparser.add_argument("--fidelity-mode", choices=("baseline", "hil"), default=None, help="Select whether the run is tagged as baseline physics mode or HIL mode (default: fidelity_mode from the params file)")


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mujoco-wheeled-uav-simulator",
        description="MuJoCo wheeled UAV simulator utilities",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
    # Defaults for a bare invocation (no subcommand -> simulate). Every option
    # read by the simulate handler must appear here or a bare run raises
    # AttributeError; test_cli.py pins this by asserting parse_args([]).
    parser.set_defaults(instance_id=0, num_uavs=1, spawn_radius=1.5, recv_port=None, send_port=None, bind_ip=None, state_target_ip=None, params_file=None, preset=None, xml_template_file=None, generated_xml_dir=None, fidelity_mode=None, pacing_mode=None, headless=False, duration_seconds=None, record=None, hold_until_first_command=None)
    subparsers = parser.add_subparsers(dest="command")

    simulate_parser = subparsers.add_parser("simulate", help="Run the MuJoCo simulator with viewer")
    check_model_parser = subparsers.add_parser("check-model", help="Render the XML/model and exit")
    hover_controller_parser = subparsers.add_parser("hover-controller", help="Run a Python hover controller that communicates over UDP")

    for subparser in (simulate_parser, check_model_parser):
        _add_model_arguments(subparser)

    simulate_parser.add_argument("--recv-port", type=int, default=None, help=f"UDP port to receive commands on (default: {PORT_RECV} + 2 * instance-id)")
    simulate_parser.add_argument("--send-port", type=int, default=None, help=f"UDP port to send state on (default: {PORT_SEND} + 2 * instance-id)")
    simulate_parser.add_argument("--bind-ip", default=None, help="Local IP address to bind the simulator command socket to (default: 127.0.0.1)")
    simulate_parser.add_argument("--state-target-ip", default=None, help="Destination IP address used when sending simulator state packets (default: 127.0.0.1)")
    simulate_parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer")
    simulate_parser.add_argument("--duration-seconds", type=float, default=None, help="Optional simulation duration limit in seconds")
    simulate_parser.add_argument(
        "--hold-until-first-command",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Freeze the vehicles at their spawn pose until the first control command arrives "
            "(recommended for automatic/scripted runs; presets that start resting on the ground "
            "disable it in their JSON instead). Overrides simulation.hold_until_first_command in the params file; "
            "the default vehicle_params.json enables it. No effect in lockstep pacing, where "
            "physics only advances on received commands anyway."
        ),
    )
    simulate_parser.add_argument("--record", default=None, metavar="PATH", help="Also write an offscreen render of the run to this video file (extension picks the format, e.g. .mp4 or .gif). Resolution/fps come from environment.recording in the params.")
    simulate_parser.add_argument(
        "--pacing-mode",
        choices=("realtime", "accelerated", "lockstep"),
        default=None,
        help=(
            "Simulator wall-clock pacing: realtime (default, RTF~1 for HIL/interactive), "
            "accelerated (no pacing, run as fast as CPU allows), or lockstep "
            "(deterministic co-simulation: block until a control command arrives "
            "before advancing each control period). "
            "Overrides simulation.pacing_mode in vehicle_params.json when set."
        ),
    )

    hover_controller_parser.add_argument("--instance-id", type=int, default=0, help="Controller instance id used to derive default ports")
    hover_controller_parser.add_argument("--bind-ip", default="127.0.0.1", help="Local IP address to bind the controller state-receive socket to")
    hover_controller_parser.add_argument("--target-ip", default="127.0.0.1", help="Destination IP address used when sending control commands to the simulator")
    hover_controller_parser.add_argument("--local-port", type=int, default=None, help=f"UDP port to receive simulator state on (default: {PORT_SEND} + 2 * instance-id)")
    hover_controller_parser.add_argument("--target-port", type=int, default=None, help=f"UDP port to send control commands to (default: {PORT_RECV} + 2 * instance-id)")
    hover_controller_parser.add_argument("--params-file", default=None, help="Path to a vehicle_params.json file to load instead of the repository default")
    hover_controller_parser.add_argument("--target-position", nargs=3, type=float, metavar=("X", "Y", "Z"), default=[0.0, 0.0, 1.5], help="Hover target position in world coordinates")
    hover_controller_parser.add_argument("--fidelity-mode", choices=("baseline", "hil"), default=None, help="Tag outgoing controller packets as baseline physics mode or HIL mode (default: fidelity_mode from the params file)")
    hover_controller_parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Stop after this many simulation seconds (state.time elapsed). Works with both realtime and accelerated simulator pacing.",
    )
    hover_controller_parser.add_argument("--state-timeout-seconds", type=float, default=10.0, help="Maximum wall-clock time to wait for state packets before stopping")
    hover_controller_parser.add_argument("--status-display-interval", type=float, default=2.0, help="Interval for controller status printouts in simulation seconds")

    return parser


def _validate_arguments(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    if arguments.instance_id < 0:
        parser.error("--instance-id must be non-negative")
    if arguments.num_uavs <= 0:
        parser.error("--num-uavs must be positive")
    if arguments.spawn_radius <= 0.0:
        parser.error("--spawn-radius must be positive")
    if arguments.duration_seconds is not None and arguments.duration_seconds <= 0.0:
        parser.error("--duration-seconds must be positive")
    if getattr(arguments, "state_timeout_seconds", 1.0) <= 0.0:
        parser.error("--state-timeout-seconds must be positive")
    if getattr(arguments, "status_display_interval", 1.0) <= 0.0:
        parser.error("--status-display-interval must be positive")


def main(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    arguments = parser.parse_args(argv)
    command = arguments.command or "simulate"
    _validate_arguments(parser, arguments)

    if command == "simulate":
        from .runtime import SimulationRequest, run_simulation

        run_simulation(
            SimulationRequest(
                instance_id=arguments.instance_id,
                recv_port=arguments.recv_port,
                send_port=arguments.send_port,
                bind_ip=arguments.bind_ip or "127.0.0.1",
                state_target_ip=arguments.state_target_ip or "127.0.0.1",
                num_uavs=arguments.num_uavs,
                spawn_radius=arguments.spawn_radius,
                params_path=resolve_params_file(arguments.params_file, arguments.preset),
                template_path=arguments.xml_template_file,
                generated_xml_dir=arguments.generated_xml_dir,
                fidelity_mode=arguments.fidelity_mode,
                pacing_mode=arguments.pacing_mode,
                headless=arguments.headless,
                duration_seconds=arguments.duration_seconds,
                record_path=arguments.record,
                hold_until_first_command=arguments.hold_until_first_command,
            )
        )
        return 0
    if command == "check-model":
        from .runtime import SimulationRequest, validate_model

        return validate_model(
            SimulationRequest(
                instance_id=arguments.instance_id,
                num_uavs=arguments.num_uavs,
                spawn_radius=arguments.spawn_radius,
                params_path=resolve_params_file(arguments.params_file, arguments.preset),
                template_path=arguments.xml_template_file,
                generated_xml_dir=arguments.generated_xml_dir,
                fidelity_mode=arguments.fidelity_mode,
            )
        )
    if command == "hover-controller":
        from .controllers.hover import run_hover_controller

        run_hover_controller(
            instance_id=arguments.instance_id,
            bind_ip=arguments.bind_ip,
            target_ip=arguments.target_ip,
            local_port=arguments.local_port,
            target_port=arguments.target_port,
            params_path=arguments.params_file,
            target_position=arguments.target_position,
            duration_seconds=arguments.duration_seconds,
            state_timeout_seconds=arguments.state_timeout_seconds,
            status_display_interval=arguments.status_display_interval,
            fidelity_mode=arguments.fidelity_mode,
        )
        return 0

    parser.error(f"Unsupported command: {command}")
    return 2


if __name__ == "__main__":
    # `python -m wheeled_uav.cli` is how the MATLAB side launches the simulator
    # (uavsim.Launch, whenever a .venv interpreter is found). Without this the
    # module merely imports and exits 0 in silence, so the launcher sees a
    # successful start, no process on the UDP port, and blames the timeout on
    # a slow simulator.
    raise SystemExit(main())
