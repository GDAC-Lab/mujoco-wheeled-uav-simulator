function wall_demo_controller_impl(varargin)
% WALL_DEMO_CONTROLLER_IMPL  Minimal wall-riding demo over UDP (single UAV).
%
% The most obvious way to press a wheeled vehicle against a wall: run the
% UNMODIFIED shared hover controller and simply place its target position
% BEHIND the wall face. The position error keeps pulling the vehicle into
% the wall, the wall's reaction stops it, and the height component of the
% same target follows a scripted climb/hold/descend profile while the
% wheels roll on the wall. There is no wall-specific control law at all --
% every sample is the exact same uavsim.Control.compute_hover_control call
% the hovering sample uses. The point is only to demonstrate that the
% simulator supports wheel-on-wall scenarios end to end (spawn, contact
% reporting, logging).
%
% Name/value options beyond the shared uavsim.RunOptions set:
%   'press_depth_m' how far behind the wall-contact point the target sits
%                   while pressed (default 1.0 m; steady pressing force is
%                   roughly position_gain_x * press_depth_m)
%   'z_low'         profile start/end height (default 0.6 m)
%   'z_high'        profile top height (default 1.6 m)
%   'contact_x'     world x where the wheels meet the wall face. Default
%                   matches configs/vehicle_params.wall_demo.json:
%                   wall face 1.95 m - wheel radius 0.15 m = 1.80 m.
close all; clc;

runtime_options = uavsim.RunOptions.parse(varargin, { ...
    'press_depth_m', 1.0, @(value) isnumeric(value) && isscalar(value) && value > 0.0; ...
    'z_low', 0.6, @(value) isnumeric(value) && isscalar(value) && value > 0.0; ...
    'z_high', 1.6, @(value) isnumeric(value) && isscalar(value) && value > 0.0; ...
    'contact_x', 1.80, @(value) isnumeric(value) && isscalar(value) ...
});
project_directory = fileparts(fileparts(fileparts(mfilename('fullpath'))));
controller_session = uavsim.Session.start( ...
    project_directory, ...
    runtime_options, ...
    'simulator_root', uavsim.Util.resolve_path_option(runtime_options.simulator_root, project_directory), ...
    'params_path', uavsim.Util.resolve_path_option(runtime_options.params_path, fullfile(project_directory, 'configs', 'vehicle_params.wall_demo.json')), ...
    'generated_xml_directory', uavsim.Util.resolve_path_option(runtime_options.generated_xml_directory, fullfile(project_directory, 'build', 'generated_xml')) ...
);
vehicle_params = controller_session.vehicle_params;
instance_options = controller_session.instance_options;
controller_socket = controller_session.controller_socket;
socket_cleanup_handler = onCleanup(@() uavsim.Session.cleanup_socket(controller_socket));
simulator_cleanup_handler = onCleanup(@() uavsim.Launch.cleanup_simulator_process(controller_session.simulator_process_id, controller_session.simulator_options));

controller_config = vehicle_params.controller;
[allocation_matrix, mixer] = uavsim.Params.build_allocation_and_mixer(vehicle_params);
command_options = uavsim.Protocol.build_command_options(vehicle_params.command_mode, vehicle_params.thrust_coefficient, 'fidelity_mode', vehicle_params.fidelity.mode);
logging_options = uavsim.RunOptions.build_logging_options('wall_demo', instance_options);
runtime_metrics = uavsim.Metrics.initialize();

logger_config = build_logger_config(vehicle_params, controller_config, runtime_options, allocation_matrix, mixer, command_options, instance_options);
logger = simulation_logger(project_directory, logger_config, logging_options);
% Initialized lazily from the first sample's absolute sim time: attaching to
% a simulator already at t >> interval must not trigger a save storm.
next_log_save_time = NaN;
cleanup_handler = onCleanup(@() uavsim.Session.finalize_controller_run(logger));

status_display_interval = 2.0;
next_status_time = 0.0;
start_time = NaN;
idle_deadline = tic;
udp_sample_tracker = [];

pressed_sample_count = 0;
pressed_contact_count = 0;
peak_height = -inf;

fprintf('Wall demo controller started (%s). Profile: approach -> engage -> climb -> hold -> descend -> release.\n', ...
    instance_options.label);
uavsim.Session.display_logging_behavior(logger);

try
    while true
        state = uavsim.Protocol.read_latest_state(controller_socket);
        if isempty(state)
            if toc(idle_deadline) >= runtime_options.state_timeout_seconds
                runtime_metrics = uavsim.Metrics.note_timeout(runtime_metrics); %#ok<NASGU>
                error('uavsim:stateTimeout', 'No simulator state received within %.1f s.', runtime_options.state_timeout_seconds);
            end
            pause(0.001);
            continue;
        end

        idle_deadline = tic;
        uavsim.Protocol.assert_single_uav_state(state, 'wall_demo_controller');

        % One control evaluation per new simulator sample (docs/TIMING.md).
        [is_new_sample, udp_sample_tracker] = uavsim.Protocol.udp_state_is_new(state, udp_sample_tracker);
        if ~is_new_sample
            pause(0.0002);
            continue;
        end

        if isnan(start_time)
            start_time = double(state.time);
        end

        elapsed_simulation_time = double(state.time) - start_time;
        [phase_name, target_position, is_pressed, profile_done] = demo_profile(elapsed_simulation_time, runtime_options);
        if profile_done || (isfinite(runtime_options.duration_seconds) && elapsed_simulation_time >= runtime_options.duration_seconds)
            fprintf('Wall demo complete at t=%.2f s (peak height %.2f m).\n', elapsed_simulation_time, peak_height);
            break;
        end

        % The demo has no control law of its own: this is the exact call the
        % hovering sample makes. Only the target position moves.
        compute_timer = tic;
        rotor_thrusts = uavsim.Control.compute_hover_control( ...
            state, ...
            target_position, ...
            controller_config.desired_heading, ...
            vehicle_params.mass, ...
            vehicle_params.gravity, ...
            controller_config.position_gain, ...
            controller_config.velocity_gain, ...
            controller_config.attitude_gain, ...
            controller_config.angular_velocity_gain, ...
            mixer, ...
            vehicle_params.max_rotor_thrust, ...
            controller_config.position_error_limit_m, ...
            controller_config.max_tilt_deg ...
        );
        runtime_metrics = uavsim.Metrics.update(runtime_metrics, state, toc(compute_timer) * 1.0e3);

        control_command = uavsim.Protocol.build_control_command( ...
            rotor_thrusts, ...
            command_options, ...
            'sequence', runtime_metrics.command_sequence, ...
            'source_state_sequence', runtime_metrics.last_source_state_sequence, ...
            'wall_time_send_ns', uavsim.Util.wall_time_now_ns(), ...
            'controller_compute_ms', runtime_metrics.last_controller_compute_ms, ...
            'state_age_ms', runtime_metrics.last_state_age_ms, ...
            'state_sequence_gap', runtime_metrics.last_state_sequence_gap ...
        );
        uavsim.Protocol.send_control_command(controller_socket, control_command, controller_session.target_ip, controller_session.target_port);

        logger.append(state, control_command, target_position);

        if isnan(next_log_save_time)
            next_log_save_time = double(state.time) + logging_options.periodic_interval_seconds;
        end
        if uavsim.RunOptions.should_save_log_periodically(logging_options, state.time, next_log_save_time)
            logger.save_snapshot(double(state.time), 'periodic');
            if logging_options.print_save_events
                fprintf('Simulation log checkpoint saved at t=%.2f s -> %s\n', double(state.time), logger.get_file_path());
            end
            next_log_save_time = double(state.time) + logging_options.periodic_interval_seconds;
        end

        wall_normal_force = read_wall_normal_force(state);
        if is_pressed
            pressed_sample_count = pressed_sample_count + 1;
            if wall_normal_force > 0.1
                pressed_contact_count = pressed_contact_count + 1;
            end
        end
        position = uavsim.Util.state_vector(state.position);
        peak_height = max(peak_height, position(3));

        if state.time >= next_status_time
            display_status(state, phase_name, target_position, wall_normal_force, instance_options.label);
            next_status_time = state.time + status_display_interval;
        end
    end

    if pressed_sample_count > 0
        fprintf('Wall contact while pressed: %.1f%% of %d samples.\n', ...
            100.0 * pressed_contact_count / pressed_sample_count, pressed_sample_count);
    end
catch execution_error
    if strcmp(execution_error.identifier, 'uavsim:stateTimeout')
        fprintf('\nController stopped: %s\n', execution_error.message);
        return;
    end
    rethrow(execution_error);
end
end


function [phase_name, target_position, is_pressed, profile_done] = demo_profile(elapsed_seconds, runtime_options)
% Scripted waypoint profile. While "pressed" the target x sits press_depth_m
% BEHIND the wall-contact point (the wall stops the vehicle, the position
% error provides the press); y stays on the wall center line and only the
% height setpoint moves. The target slides in/out over a short interval so
% engagement is bounce-free.
z_low = runtime_options.z_low;
z_high = runtime_options.z_high;
contact_x = runtime_options.contact_x;
press_depth = runtime_options.press_depth_m;
approach_x = contact_x - 0.05;
pressed_x = contact_x + press_depth;
slide_seconds = 1.5;

approach_end = 5.0;
engage_end = 8.0;
climb_end = 16.0;
hold_end = 20.0;
descend_end = 28.0;
release_end = 31.0;
retreat_end = 34.0;

profile_done = false;

if elapsed_seconds < approach_end
    phase_name = 'approach';
    target_position = [approach_x; 0.0; z_low];
elseif elapsed_seconds < engage_end
    phase_name = 'engage';
    slide_progress = min(1.0, (elapsed_seconds - approach_end) / slide_seconds);
    target_position = [approach_x + slide_progress * (pressed_x - approach_x); 0.0; z_low];
elseif elapsed_seconds < climb_end
    phase_name = 'climb';
    climb_progress = (elapsed_seconds - engage_end) / (climb_end - engage_end);
    target_position = [pressed_x; 0.0; z_low + (z_high - z_low) * climb_progress];
elseif elapsed_seconds < hold_end
    phase_name = 'hold';
    target_position = [pressed_x; 0.0; z_high];
elseif elapsed_seconds < descend_end
    phase_name = 'descend';
    descend_progress = (elapsed_seconds - hold_end) / (descend_end - hold_end);
    target_position = [pressed_x; 0.0; z_high - (z_high - z_low) * descend_progress];
elseif elapsed_seconds < release_end
    phase_name = 'release';
    slide_progress = min(1.0, (elapsed_seconds - descend_end) / slide_seconds);
    target_position = [pressed_x + slide_progress * (approach_x - pressed_x); 0.0; z_low];
elseif elapsed_seconds < retreat_end
    phase_name = 'retreat';
    target_position = [contact_x - 0.25; 0.0; z_low];
else
    phase_name = 'done';
    target_position = [contact_x - 0.25; 0.0; z_low];
    profile_done = true;
end

% "Pressed" bookkeeping: count contact statistics only while the target is
% (mostly) behind the wall face.
is_pressed = target_position(1) > contact_x + 0.5 * press_depth;
end


function wall_normal_force = read_wall_normal_force(state)
% Wall-group contact summary published by the simulator; 0.0 when absent.
wall_normal_force = 0.0;
if isfield(state, 'contact_summary') && isstruct(state.contact_summary) ...
        && isfield(state.contact_summary, 'wall') && isstruct(state.contact_summary.wall) ...
        && isfield(state.contact_summary.wall, 'total_normal_force')
    wall_normal_force = double(state.contact_summary.wall.total_normal_force);
end
end


function display_status(state, phase_name, target_position, wall_normal_force, instance_label)
position = reshape(double(state.position), [], 1);
realtime_factor = uavsim.Protocol.get_realtime_factor(state);
fprintf( ...
    '[%s t=%.2f s, rtf=%.2f, %s] pos=[%.3f %.3f %.3f] m, target=[%.2f %.2f] (x z) m, wall_N=%.2f N\n', ...
    instance_label, ...
    state.time, ...
    realtime_factor, ...
    phase_name, ...
    position(1), position(2), position(3), ...
    target_position(1), target_position(3), ...
    wall_normal_force ...
);
end


function config = build_logger_config(vehicle_params, controller_config, runtime_options, allocation_matrix, mixer, command_options, instance_options)
config = struct( ...
    'controller', 'wall_demo_controller', ...
    'mass', vehicle_params.mass, ...
    'gravity', vehicle_params.gravity, ...
    'arm_x', vehicle_params.arm_x, ...
    'arm_y', vehicle_params.arm_y, ...
    'yaw_moment_ratio', vehicle_params.yaw_moment_ratio, ...
    'max_rotor_thrust', vehicle_params.max_rotor_thrust, ...
    'thrust_coefficient', vehicle_params.thrust_coefficient, ...
    'rotor_geometry', vehicle_params.rotors, ...
    'position_gain', controller_config.position_gain, ...
    'velocity_gain', controller_config.velocity_gain, ...
    'attitude_gain', controller_config.attitude_gain, ...
    'angular_velocity_gain', controller_config.angular_velocity_gain, ...
    'position_error_limit_m', controller_config.position_error_limit_m, ...
    'max_tilt_deg', controller_config.max_tilt_deg, ...
    'desired_heading', controller_config.desired_heading, ...
    'press_depth_m', runtime_options.press_depth_m, ...
    'z_low', runtime_options.z_low, ...
    'z_high', runtime_options.z_high, ...
    'contact_x', runtime_options.contact_x, ...
    'allocation_matrix', allocation_matrix, ...
    'mixer', mixer, ...
    'command_mode', command_options.input_mode, ...
    'fidelity_mode', vehicle_params.fidelity.mode, ...
    'network_fidelity', vehicle_params.fidelity.network, ...
    'instance_id', instance_options.instance_id, ...
    'instance_label', instance_options.label ...
);
end
