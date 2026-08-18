function multi_uav_formation_controller_impl(varargin)
% MULTI_UAV_FORMATION_CONTROLLER_IMPL  Circular formation in one MuJoCo world.
%
% Pairs with `simulate --num-uavs N`. Reads batched multi-UAV state packets,
% computes a centroid + slot-error hover target per UAV, and sends one batched
% command datagram. Defaults come from vehicle_params.json:formation.
%
% Name/value options beyond the shared uavsim.RunOptions set (each overrides
% the vehicle_params.json:formation default):
%   'num_uavs'               UAV count (must match simulate --num-uavs)
%   'formation_radius'       circular slot radius [m]
%   'spawn_radius'           spawn ring radius forwarded to auto-launch [m]
%   'base_height'            formation altitude [m]
%   'centroid_target_xy'     [x y] centroid target [m]
%   'centroid_gain'          centroid error -> target shift gain
%   'formation_gain'         slot error -> target shift gain
%   'desired_heading', 'position_gain', 'velocity_gain', 'attitude_gain',
%   'angular_velocity_gain'  hover-law vectors (see README controller section)
%   'idle_sleep_seconds'     poll sleep while waiting for packets [s]
%   'status_display_interval' status print period [sim s]
%   'target_ip'              simulator command destination IP
%   'formation_log_mode'     'bundle_only' (default) | 'bundle_and_individual'
close all; clc;

runtime_options = uavsim.RunOptions.parse(varargin, { ...
    'num_uavs', [], @(value) isempty(value) || (isnumeric(value) && isscalar(value)); ...
    'formation_radius', [], @(value) isempty(value) || (isnumeric(value) && isscalar(value)); ...
    'spawn_radius', [], @(value) isempty(value) || (isnumeric(value) && isscalar(value)); ...
    'base_height', [], @(value) isempty(value) || (isnumeric(value) && isscalar(value)); ...
    'centroid_target_xy', [], @(value) isempty(value) || (isnumeric(value) && numel(value) == 2); ...
    'centroid_gain', [], @(value) isempty(value) || (isnumeric(value) && isscalar(value)); ...
    'formation_gain', [], @(value) isempty(value) || (isnumeric(value) && isscalar(value)); ...
    'idle_sleep_seconds', [], @(value) isempty(value) || (isnumeric(value) && isscalar(value)); ...
    'status_display_interval', [], @(value) isempty(value) || (isnumeric(value) && isscalar(value)); ...
    'desired_heading', [], @(value) isempty(value) || (isnumeric(value) && numel(value) == 3); ...
    'position_gain', [], @(value) isempty(value) || (isnumeric(value) && numel(value) == 3); ...
    'velocity_gain', [], @(value) isempty(value) || (isnumeric(value) && numel(value) == 3); ...
    'attitude_gain', [], @(value) isempty(value) || (isnumeric(value) && numel(value) == 3); ...
    'angular_velocity_gain', [], @(value) isempty(value) || (isnumeric(value) && numel(value) == 3); ...
    'target_ip', '127.0.0.1', @(value) ischar(value) || (isstring(value) && isscalar(value)); ...
    'formation_log_mode', 'bundle_only', @(value) any(strcmp(char(value), {'bundle_and_individual', 'bundle_only'})) ...
});
runtime_options.target_ip = char(runtime_options.target_ip);
runtime_options.formation_log_mode = char(runtime_options.formation_log_mode);

project_directory = fileparts(fileparts(fileparts(mfilename('fullpath'))));
params_path = uavsim.Util.resolve_path_option(runtime_options.params_path, fullfile(project_directory, 'vehicle_params.json'));
vehicle_params = uavsim.Params.load(project_directory, 'params_path', params_path);
formation_config = apply_runtime_defaults(runtime_options, vehicle_params.formation);
num_uavs = formation_config.num_uavs;
desired_offsets_xy = build_circular_offsets(num_uavs, formation_config.formation_radius);
target_centroid_xy = reshape(double(formation_config.centroid_target_xy), 2, 1);

runtime_options.num_uavs = num_uavs;
runtime_options.spawn_radius = formation_config.spawn_radius;
if ~isfinite(runtime_options.duration_seconds)
    runtime_options.duration_seconds = formation_config.duration_seconds;
end
controller_session = uavsim.Session.start( ...
    project_directory, ...
    runtime_options, ...
    'target_ip', runtime_options.target_ip, ...
    'simulator_root', uavsim.Util.resolve_path_option(runtime_options.simulator_root, project_directory), ...
    'params_path', params_path, ...
    'generated_xml_directory', uavsim.Util.resolve_path_option(runtime_options.generated_xml_directory, fullfile(project_directory, 'build', 'generated_xml')) ...
);
instance_options = controller_session.instance_options;
controller_socket = controller_session.controller_socket;
socket_cleanup_handler = onCleanup(@() uavsim.Session.cleanup_socket(controller_socket));
simulator_cleanup_handler = onCleanup(@() uavsim.Launch.cleanup_simulator_process(controller_session.simulator_process_id, controller_session.simulator_options));

[allocation_matrix, mixer] = uavsim.Params.build_allocation_and_mixer(vehicle_params);
command_options = uavsim.Protocol.build_command_options(vehicle_params.command_mode, vehicle_params.thrust_coefficient, 'fidelity_mode', vehicle_params.fidelity.mode);
runtime_metrics = uavsim.Metrics.initialize();

loggers = build_loggers(project_directory, instance_options, vehicle_params, formation_config, ...
    desired_offsets_xy, target_centroid_xy, allocation_matrix, mixer, command_options, runtime_options.formation_log_mode);
logger_cleanup_handler = onCleanup(@() finalize_formation_logging(loggers, project_directory, instance_options));

status_display_interval = formation_config.status_display_interval;
next_status_time = 0.0;
start_time = NaN;
idle_deadline = tic;
udp_sample_tracker = [];

fprintf('Starting multi-UAV formation controller for %d UAVs.\n', num_uavs);
fprintf('Simulator routing -> %s, recv=%d, send=%d\n', instance_options.label, instance_options.controller_local_port, controller_session.target_port);
fprintf('Formation radius: %.2f m, spawn radius: %.2f m, base height: %.2f m, centroid target=[%.2f %.2f] m\n', ...
    formation_config.formation_radius, formation_config.spawn_radius, formation_config.base_height, target_centroid_xy(1), target_centroid_xy(2));

try
    while true
        state_packet = uavsim.Protocol.read_latest_state(controller_socket);
        if isempty(state_packet) || ~isfield(state_packet, 'uavs')
            if toc(idle_deadline) >= runtime_options.state_timeout_seconds
                runtime_metrics = uavsim.Metrics.note_timeout(runtime_metrics); %#ok<NASGU>
                error('uavsim:stateTimeout', 'No simulator state received within %.1f s.', runtime_options.state_timeout_seconds);
            end
            pause(formation_config.idle_sleep_seconds);
            continue;
        end

        idle_deadline = tic;

        [is_new_sample, udp_sample_tracker] = uavsim.Protocol.udp_state_is_new(state_packet, udp_sample_tracker);
        if ~is_new_sample
            pause(formation_config.idle_sleep_seconds);
            continue;
        end

        % Cell-safe normalization (jsondecode can yield struct OR cell arrays).
        states = uavsim.Protocol.uav_state_list(state_packet);
        if numel(states) ~= num_uavs
            % A silent pause-loop here can never time out (a packet DID
            % arrive), so a pairing mistake must fail fast and loudly.
            error('uavsim:uavCountMismatch', ...
                'Simulator sent %d UAV states but the controller expects %d. Launch the simulator with `simulate --num-uavs %d` or adjust the num_uavs option.', ...
                numel(states), num_uavs, num_uavs);
        end

        simulation_time = double(state_packet.time);
        if isnan(start_time)
            start_time = simulation_time;
        end
        elapsed_simulation_time = simulation_time - start_time;
        if isfinite(runtime_options.duration_seconds) && elapsed_simulation_time >= runtime_options.duration_seconds
            fprintf('Formation run complete at t=%.2f s\n', elapsed_simulation_time);
            break;
        end

        positions_xy = gather_xy_positions(states);
        centroid_xy = mean(positions_xy, 2);
        centroid_error_xy = target_centroid_xy - centroid_xy;
        max_slot_error = 0.0;
        realtime_factors = zeros(num_uavs, 1);
        control_commands = cell(num_uavs, 1);
        target_positions = cell(num_uavs, 1);
        rotor_thrusts_by_uav = cell(num_uavs, 1);
        state_metrics = uavsim.Protocol.get_state_packet_metrics(states{1});
        command_wall_time_ns = uavsim.Util.wall_time_now_ns();

        compute_timer = tic;
        for uav_index = 1:num_uavs
            state = states{uav_index};
            current_position = reshape(double(state.position), [], 1);
            current_relative_xy = current_position(1:2) - centroid_xy;
            slot_error_xy = desired_offsets_xy(:, uav_index) - current_relative_xy;
            target_xy = current_position(1:2) ...
                + formation_config.centroid_gain * centroid_error_xy ...
                + formation_config.formation_gain * slot_error_xy;
            target_positions{uav_index} = [target_xy; formation_config.base_height];

            rotor_thrusts_by_uav{uav_index} = uavsim.Control.compute_hover_control( ...
                state, ...
                target_positions{uav_index}, ...
                formation_config.desired_heading, ...
                vehicle_params.mass, ...
                vehicle_params.gravity, ...
                formation_config.position_gain, ...
                formation_config.velocity_gain, ...
                formation_config.attitude_gain, ...
                formation_config.angular_velocity_gain, ...
                mixer, ...
                vehicle_params.max_rotor_thrust, ...
                formation_config.position_error_limit_m, ...
                formation_config.max_tilt_deg ...
            );
            realtime_factors(uav_index) = uavsim.Protocol.get_realtime_factor(state);
            max_slot_error = max(max_slot_error, norm(slot_error_xy));
        end
        % One metrics update per simulator sample: only one datagram is sent,
        % so the command sequence must advance by one, and the state-gap
        % accounting must not be zeroed by re-reading the same sequence N
        % times (all UAVs share the packet-level metadata of states{1}).
        runtime_metrics = uavsim.Metrics.update(runtime_metrics, states{1}, toc(compute_timer) * 1.0e3);

        for uav_index = 1:num_uavs
            control_command = uavsim.Protocol.build_control_command( ...
                rotor_thrusts_by_uav{uav_index}, ...
                command_options, ...
                'sequence', runtime_metrics.command_sequence, ...
                'source_state_sequence', state_metrics.sequence, ...
                'wall_time_send_ns', command_wall_time_ns, ...
                'controller_compute_ms', runtime_metrics.last_controller_compute_ms, ...
                'state_age_ms', runtime_metrics.last_state_age_ms, ...
                'state_sequence_gap', runtime_metrics.last_state_sequence_gap ...
            );
            control_commands{uav_index} = control_command;
            loggers{uav_index}.append(states{uav_index}, control_command, target_positions{uav_index});
        end

        uavsim.Protocol.send_multi_uav_control_command(controller_socket, control_commands, command_options, controller_session.target_ip, controller_session.target_port);

        if simulation_time >= next_status_time
            display_status(elapsed_simulation_time, centroid_xy, centroid_error_xy, max_slot_error, realtime_factors, num_uavs);
            next_status_time = simulation_time + status_display_interval;
        end

        pause(formation_config.idle_sleep_seconds);
    end
catch execution_error
    if strcmp(execution_error.identifier, 'uavsim:stateTimeout')
        fprintf('\nMulti-UAV formation controller stopped: %s\n', execution_error.message);
        return;
    end
    rethrow(execution_error);
end
end


function formation_config = apply_runtime_defaults(runtime_options, formation_defaults)
% Merge explicit name/value options over the vehicle_params.json defaults.
formation_config = formation_defaults;
formation_config.num_uavs = resolve_scalar_option(runtime_options.num_uavs, formation_defaults.num_uavs);
formation_config.formation_radius = resolve_scalar_option(runtime_options.formation_radius, formation_defaults.formation_radius);
formation_config.spawn_radius = resolve_scalar_option(runtime_options.spawn_radius, formation_defaults.spawn_radius);
formation_config.base_height = resolve_scalar_option(runtime_options.base_height, formation_defaults.base_height);
formation_config.centroid_gain = resolve_scalar_option(runtime_options.centroid_gain, formation_defaults.centroid_gain);
formation_config.formation_gain = resolve_scalar_option(runtime_options.formation_gain, formation_defaults.formation_gain);
formation_config.idle_sleep_seconds = resolve_scalar_option(runtime_options.idle_sleep_seconds, formation_defaults.idle_sleep_seconds);
formation_config.status_display_interval = resolve_scalar_option(runtime_options.status_display_interval, formation_defaults.status_display_interval);
formation_config.centroid_target_xy = resolve_vector_option(runtime_options.centroid_target_xy, formation_defaults.centroid_target_xy, 2);
formation_config.desired_heading = resolve_vector_option(runtime_options.desired_heading, formation_defaults.desired_heading, 3);
formation_config.position_gain = resolve_vector_option(runtime_options.position_gain, formation_defaults.position_gain, 3);
formation_config.velocity_gain = resolve_vector_option(runtime_options.velocity_gain, formation_defaults.velocity_gain, 3);
formation_config.attitude_gain = resolve_vector_option(runtime_options.attitude_gain, formation_defaults.attitude_gain, 3);
formation_config.angular_velocity_gain = resolve_vector_option(runtime_options.angular_velocity_gain, formation_defaults.angular_velocity_gain, 3);
end


function value = resolve_scalar_option(value, default_value)
if isempty(value)
    value = default_value;
else
    value = double(value);
end
end


function value = resolve_vector_option(value, default_value, expected_length)
if isempty(value)
    value = reshape(double(default_value), [], 1);
else
    value = reshape(double(value), [], 1);
end
if numel(value) ~= expected_length
    error('uavsim:invalidVector', 'Expected vector of length %d.', expected_length);
end
end


function desired_offsets_xy = build_circular_offsets(num_uavs, formation_radius)
angles = 2.0 * pi * (0:(num_uavs - 1)) / num_uavs;
desired_offsets_xy = formation_radius * [cos(angles); sin(angles)];
end


function positions_xy = gather_xy_positions(states)
num_uavs = numel(states);
positions_xy = zeros(2, num_uavs);
for uav_index = 1:num_uavs
    position = reshape(double(states{uav_index}.position), [], 1);
    positions_xy(:, uav_index) = position(1:2);
end
end


function loggers = build_loggers(project_directory, instance_options, vehicle_params, formation_config, ...
    desired_offsets_xy, target_centroid_xy, allocation_matrix, mixer, command_options, formation_log_mode)
num_uavs = formation_config.num_uavs;
loggers = cell(num_uavs, 1);
for uav_index = 1:num_uavs
    logging_options = uavsim.RunOptions.build_logging_options( ...
        sprintf('formation_uav_%d', uav_index), instance_options, ...
        'print_save_events', ~strcmp(formation_log_mode, 'bundle_only'));
    logging_options.formation_log_mode = formation_log_mode;

    config = struct( ...
        'controller', 'multi_uav_formation_controller', ...
        'mass', vehicle_params.mass, ...
        'gravity', vehicle_params.gravity, ...
        'max_rotor_thrust', vehicle_params.max_rotor_thrust, ...
        'thrust_coefficient', vehicle_params.thrust_coefficient, ...
        'rotor_geometry', vehicle_params.rotors, ...
        'position_gain', formation_config.position_gain, ...
        'velocity_gain', formation_config.velocity_gain, ...
        'attitude_gain', formation_config.attitude_gain, ...
        'angular_velocity_gain', formation_config.angular_velocity_gain, ...
        'position_error_limit_m', formation_config.position_error_limit_m, ...
        'max_tilt_deg', formation_config.max_tilt_deg, ...
        'desired_heading', formation_config.desired_heading, ...
        'allocation_matrix', allocation_matrix, ...
        'mixer', mixer, ...
        'command_mode', command_options.input_mode, ...
        'fidelity_mode', vehicle_params.fidelity.mode, ...
        'instance_id', instance_options.instance_id, ...
        'instance_label', instance_options.label, ...
        'uav_index', uav_index, ...
        'num_uavs', num_uavs, ...
        'formation_radius', formation_config.formation_radius, ...
        'spawn_radius', formation_config.spawn_radius, ...
        'base_height', formation_config.base_height, ...
        'centroid_target_xy', target_centroid_xy, ...
        'desired_offset_xy', desired_offsets_xy(:, uav_index), ...
        'centroid_gain', formation_config.centroid_gain, ...
        'formation_gain', formation_config.formation_gain ...
    );

    loggers{uav_index} = simulation_logger(project_directory, config, logging_options);
end
end


function finalize_formation_logging(loggers, project_directory, instance_options)
for uav_index = 1:numel(loggers)
    if isempty(loggers{uav_index})
        continue;
    end
    uavsim.Session.finalize_controller_run(loggers{uav_index});
end

write_combined_formation_log(loggers, project_directory, instance_options);

if strcmp(resolve_formation_log_mode(loggers), 'bundle_only')
    delete_individual_formation_logs(loggers);
end
end


function write_combined_formation_log(loggers, project_directory, instance_options)
log_paths = cellfun(@(logger) char(logger.get_file_path()), loggers, 'UniformOutput', false);
existing_mask = cellfun(@isfile, log_paths);
log_paths = log_paths(existing_mask);
if isempty(log_paths)
    return;
end

loaded_logs = cell(numel(log_paths), 1);
for log_index = 1:numel(log_paths)
    loaded_log = uavsim.LogFiles.load_log(log_paths{log_index});
    loaded_log.source_path = log_paths{log_index};
    loaded_logs{log_index} = loaded_log;
end

uav_indices = cellfun(@(log_entry) double(log_entry.config.uav_index), loaded_logs);
[~, sort_index] = sort(uav_indices);
loaded_logs = loaded_logs(sort_index);
log_paths = log_paths(sort_index);

timestamp = uavsim.LogFiles.extract_timestamp(log_paths{1});
if isempty(timestamp)
    timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
end

formation_log = struct();
formation_log.meta = struct( ...
    'format_version', 1, ...
    'created_at', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), ...
    'num_uavs', numel(loaded_logs), ...
    'instance_id', instance_options.instance_id, ...
    'instance_label', instance_options.label, ...
    'timestamp', timestamp, ...
    'bundle_layout', 'logs_and_named_uavs' ...
);
formation_log.logs = {loaded_logs};
formation_log.source_paths = {log_paths};
formation_log.uavs = build_named_uav_log_struct(loaded_logs);

log_directory = fullfile(project_directory, 'logs');
bundle_path = fullfile(log_directory, sprintf('formation_bundle%s_%s.mat', instance_options.file_suffix, timestamp));
save(bundle_path, 'formation_log');
fprintf('Combined formation log saved -> %s\n', bundle_path);
end


function named_uavs = build_named_uav_log_struct(loaded_logs)
named_uavs = struct();
for log_index = 1:numel(loaded_logs)
    field_name = sprintf('uav_%d', log_index);
    named_uavs.(field_name) = loaded_logs{log_index};
end
end


function formation_log_mode = resolve_formation_log_mode(loggers)
formation_log_mode = 'bundle_only';
if isempty(loggers) || isempty(loggers{1})
    return;
end
logger_options = loggers{1}.get_options();
if isfield(logger_options, 'formation_log_mode')
    formation_log_mode = char(logger_options.formation_log_mode);
end
end


function delete_individual_formation_logs(loggers)
for log_index = 1:numel(loggers)
    if isempty(loggers{log_index})
        continue;
    end
    file_path = char(loggers{log_index}.get_file_path());
    if isfile(file_path)
        delete(file_path);
    end
end
fprintf('Individual formation logs removed after bundle generation.\n');
end


function display_status(elapsed_simulation_time, centroid_xy, centroid_error_xy, max_slot_error, realtime_factors, num_uavs)
average_realtime_factor = mean(realtime_factors);
fprintf('[formation t=%.2f s, avg_rtf=%.2f] centroid=[%.3f %.3f] m, centroid_err=[%.3f %.3f] m, max_slot_err=%.3f m, uavs=%d\n', ...
    elapsed_simulation_time, ...
    average_realtime_factor, ...
    centroid_xy(1), centroid_xy(2), ...
    centroid_error_xy(1), centroid_error_xy(2), ...
    max_slot_error, ...
    num_uavs ...
);
end
