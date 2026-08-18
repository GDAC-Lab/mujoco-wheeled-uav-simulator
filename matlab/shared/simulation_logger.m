classdef simulation_logger < handle
    % SIMULATION_LOGGER  Buffers controller/state samples and writes .mat logs.
    %
    % The set of logged channels is defined ONCE in channel_schema(); buffer
    % allocation and growth are derived from it. Adding a channel means:
    %   1. add a schema row,
    %   2. fill it in build_sample_row() inside append(),
    %   3. place it in the output struct in write_log() (the on-disk contract).
    %
    % The saved .mat layout (variable 'log', meta.format_version = 6) is the
    % compatibility contract consumed by the review scripts; write_log() keeps
    % it assembled literally so the format stays greppable.
    %
    % Controllers can register EXTRA channels by passing an extension struct
    % to the constructor: struct('group_name', <char>, 'channels', {{name,
    % width; ...}}). The extension channels are filled from the optional
    % fifth append() argument (a status struct keyed by channel name) and are
    % written to the log as log.(group_name).(channel_name).
    properties (Access = private)
        capacity
        count
        file_path
        config
        options
        meta
        finalized
        save_count
        last_simulation_time
        buffers   % struct: one preallocated column buffer per schema channel
        schema    % instance schema: static channel_schema() + extension rows
        extension % normalized extension definition (group_name '', channels {} when unused)
    end

    properties (Constant, Access = private)
        INITIAL_CAPACITY = 5000
    end

    methods (Static, Access = private)
        function schema = channel_schema()
            % {name, width, kind}: kind 'nan' | 'zero' -> numeric buffer with
            % that fill value; 'cell' -> cell column buffer.
            rows = { ...
                'time',                                 1, 'nan'; ...
                'realtime_factor',                      1, 'nan'; ...
                'state_protocol_version',               1, 'nan'; ...
                'state_sequence',                       1, 'nan'; ...
                'state_wall_time_send_ns',              1, 'nan'; ...
                'state_age_ms',                         1, 'nan'; ...
                'state_fidelity_mode',                  1, 'cell'; ...
                'position',                             3, 'nan'; ...
                'velocity',                             3, 'nan'; ...
                'angular_velocity_body',                3, 'nan'; ...
                'angular_velocity_world',               3, 'nan'; ...
                'rotation_matrix',                      9, 'nan'; ...
                'sensor_truth_position',                3, 'nan'; ...
                'sensor_truth_velocity',                3, 'nan'; ...
                'sensor_truth_angular_velocity_body',   3, 'nan'; ...
                'sensor_truth_angular_velocity_world',  3, 'nan'; ...
                'sensor_truth_rotation_matrix',         9, 'nan'; ...
                'rotor_thrusts',                        4, 'nan'; ...
                'rotor_omega',                          4, 'nan'; ...
                'command_protocol_version',             1, 'nan'; ...
                'command_sequence',                     1, 'nan'; ...
                'command_source_state_sequence',        1, 'nan'; ...
                'command_wall_time_send_ns',            1, 'nan'; ...
                'command_fidelity_mode',                1, 'cell'; ...
                'controller_compute_ms',                1, 'nan'; ...
                'state_sequence_gap',                   1, 'nan'; ...
                'actuator_requested_rotor_thrusts',     4, 'nan'; ...
                'actuator_applied_rotor_thrusts',       4, 'nan'; ...
                'actuator_tracking_error',              4, 'nan'; ...
                'target_position',                      3, 'nan'; ...
                'contact_count',                        1, 'zero'; ...
                'total_contact_force_magnitude',        1, 'nan'; ...
                'max_contact_force_magnitude',          1, 'nan'; ...
                'total_contact_normal_force',           1, 'nan'; ...
                'max_contact_normal_force',             1, 'nan'; ...
                'left_wheel_contact_count',             1, 'zero'; ...
                'left_wheel_total_force_magnitude',     1, 'nan'; ...
                'left_wheel_total_normal_force',        1, 'nan'; ...
                'left_wheel_max_normal_force',          1, 'nan'; ...
                'right_wheel_contact_count',            1, 'zero'; ...
                'right_wheel_total_force_magnitude',    1, 'nan'; ...
                'right_wheel_total_normal_force',       1, 'nan'; ...
                'right_wheel_max_normal_force',         1, 'nan'; ...
                'surface_contact_count',                1, 'zero'; ...
                'surface_total_force_magnitude',        1, 'nan'; ...
                'surface_total_normal_force',           1, 'nan'; ...
                'surface_max_normal_force',             1, 'nan'; ...
                'wall_contact_count',                   1, 'zero'; ...
                'wall_total_force_magnitude',           1, 'nan'; ...
                'wall_total_normal_force',              1, 'nan'; ...
                'wall_max_normal_force',                1, 'nan'; ...
                'contact_details',                      1, 'cell' ...
            };
            schema = struct('name', rows(:, 1), 'width', rows(:, 2), 'kind', rows(:, 3));
        end

        function buffer = make_buffer(width, kind, capacity)
            switch kind
                case 'nan'
                    buffer = nan(capacity, width);
                case 'zero'
                    buffer = zeros(capacity, width);
                case 'cell'
                    buffer = cell(capacity, 1);
                otherwise
                    error('uavsim:badLogSchema', 'Unknown buffer kind: %s', kind);
            end
        end
    end

    methods
        function obj = simulation_logger(base_directory, config, options, extension)
            arguments
                base_directory (1, :) char
                config struct
                options struct
                extension struct = struct()
            end

            obj.capacity = obj.INITIAL_CAPACITY;
            obj.count = 0;
            obj.finalized = false;
            obj.save_count = 0;
            obj.last_simulation_time = NaN;
            obj.config = config;
            obj.options = options;
            % The controller name comes from config.controller; the fallback
            % keeps logs written by minimal configs identifiable.
            if isfield(config, 'controller')
                controller_name = char(config.controller);
            else
                controller_name = 'unknown_controller';
            end
            obj.meta = struct( ...
                'controller', controller_name, ...
                'created_at', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), ...
                'format_version', 6, ...   % v5: fixed manual channels; v6: caller-registered extension group
                'save_count', 0, ...
                'last_save_reason', '', ...
                'last_save_wall_time', '', ...
                'last_save_simulation_time', NaN ...
            );

            log_directory = fullfile(base_directory, options.directory_name);
            if ~exist(log_directory, 'dir')
                mkdir(log_directory);
            end

            timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
            obj.file_path = fullfile(log_directory, sprintf('%s_%s.mat', options.file_prefix, timestamp));

            obj.extension = normalize_extension(extension);
            obj.schema = append_extension_schema(obj.channel_schema(), obj.extension);
            obj.buffers = struct();
            for channel_index = 1:numel(obj.schema)
                channel = obj.schema(channel_index);
                obj.buffers.(channel.name) = obj.make_buffer(channel.width, channel.kind, obj.capacity);
            end
        end

        function append(obj, state, control_command, target_position, extension_status)
            % extension_status (optional): struct keyed by the extension
            % channel names registered at construction; fills the
            % log.(group_name) channels. Omitted -> NaN channels.
            if nargin < 5
                extension_status = struct();
            end
            next_index = obj.count + 1;
            obj.ensure_capacity(next_index);

            row = build_sample_row(state, control_command, target_position);
            row = add_extension_row(row, obj.extension, extension_status);
            channel_names = fieldnames(row);
            for channel_index = 1:numel(channel_names)
                channel_name = channel_names{channel_index};
                if iscell(obj.buffers.(channel_name))
                    obj.buffers.(channel_name){next_index, 1} = row.(channel_name);
                else
                    obj.buffers.(channel_name)(next_index, :) = row.(channel_name);
                end
            end

            obj.last_simulation_time = double(state.time);
            obj.count = next_index;
        end

        function save_snapshot(obj, simulation_time, reason)
            arguments
                obj
                simulation_time (1, 1) double = NaN
                reason (1, :) char = 'manual'
            end

            if obj.finalized
                return;
            end

            obj.write_log(simulation_time, reason);
        end

        function finalize(obj, simulation_time)
            arguments
                obj
                simulation_time (1, 1) double = obj.last_simulation_time
            end

            if obj.finalized
                return;
            end

            obj.write_log(simulation_time, 'finalize');
            obj.finalized = true;
        end

        function file_path = get_file_path(obj)
            file_path = obj.file_path;
        end

        function options = get_options(obj)
            options = obj.options;
        end
    end

    methods (Access = private)
        function data = filled(obj, channel_name)
            % The written rows of one channel buffer.
            buffer = obj.buffers.(channel_name);
            data = buffer(1:obj.count, :);
        end

        function write_log(obj, simulation_time, reason)
            log = struct();
            obj.save_count = obj.save_count + 1;
            obj.meta.save_count = obj.save_count;
            obj.meta.last_save_reason = reason;
            obj.meta.last_save_wall_time = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
            obj.meta.last_save_simulation_time = simulation_time;
            log.meta = obj.meta;
            log.config = obj.config;
            log.options = obj.options;
            log.state = struct( ...
                'time', obj.filled('time'), ...
                'realtime_factor', obj.filled('realtime_factor'), ...
                'protocol_version', obj.filled('state_protocol_version'), ...
                'sequence', obj.filled('state_sequence'), ...
                'wall_time_send_ns', obj.filled('state_wall_time_send_ns'), ...
                'age_ms', obj.filled('state_age_ms'), ...
                'fidelity_mode', {obj.filled('state_fidelity_mode')}, ...
                'position', obj.filled('position'), ...
                'velocity', obj.filled('velocity'), ...
                'angular_velocity_body', obj.filled('angular_velocity_body'), ...
                'angular_velocity_world', obj.filled('angular_velocity_world'), ...
                'rotation_matrix', obj.filled('rotation_matrix') ...
            );
            log.sensor_truth = struct( ...
                'position', obj.filled('sensor_truth_position'), ...
                'velocity', obj.filled('sensor_truth_velocity'), ...
                'angular_velocity_body', obj.filled('sensor_truth_angular_velocity_body'), ...
                'angular_velocity_world', obj.filled('sensor_truth_angular_velocity_world'), ...
                'rotation_matrix', obj.filled('sensor_truth_rotation_matrix') ...
            );
            log.control = struct( ...
                'command_mode', obj.config.command_mode, ...
                'rotor_thrusts', obj.filled('rotor_thrusts'), ...
                'rotor_omega', obj.filled('rotor_omega'), ...
                'protocol_version', obj.filled('command_protocol_version'), ...
                'sequence', obj.filled('command_sequence'), ...
                'source_state_sequence', obj.filled('command_source_state_sequence'), ...
                'wall_time_send_ns', obj.filled('command_wall_time_send_ns'), ...
                'fidelity_mode', {obj.filled('command_fidelity_mode')}, ...
                'controller_compute_ms', obj.filled('controller_compute_ms'), ...
                'state_sequence_gap', obj.filled('state_sequence_gap') ...
            );
            log.actuator = struct( ...
                'requested_rotor_thrusts', obj.filled('actuator_requested_rotor_thrusts'), ...
                'applied_rotor_thrusts', obj.filled('actuator_applied_rotor_thrusts'), ...
                'tracking_error', obj.filled('actuator_tracking_error') ...
            );
            log.network = struct( ...
                'state_age_ms', obj.filled('state_age_ms'), ...
                'state_sequence', obj.filled('state_sequence'), ...
                'command_sequence', obj.filled('command_sequence'), ...
                'command_source_state_sequence', obj.filled('command_source_state_sequence'), ...
                'state_sequence_gap', obj.filled('state_sequence_gap'), ...
                'controller_compute_ms', obj.filled('controller_compute_ms') ...
            );
            log.reference = struct( ...
                'target_position', obj.filled('target_position') ...
            );
            log.contact = struct( ...
                'count', obj.filled('contact_count'), ...
                'total_force_magnitude', obj.filled('total_contact_force_magnitude'), ...
                'max_force_magnitude', obj.filled('max_contact_force_magnitude'), ...
                'total_normal_force', obj.filled('total_contact_normal_force'), ...
                'max_normal_force', obj.filled('max_contact_normal_force'), ...
                'left_wheel', struct( ...
                    'count', obj.filled('left_wheel_contact_count'), ...
                    'total_force_magnitude', obj.filled('left_wheel_total_force_magnitude'), ...
                    'total_normal_force', obj.filled('left_wheel_total_normal_force'), ...
                    'max_normal_force', obj.filled('left_wheel_max_normal_force') ...
                ), ...
                'right_wheel', struct( ...
                    'count', obj.filled('right_wheel_contact_count'), ...
                    'total_force_magnitude', obj.filled('right_wheel_total_force_magnitude'), ...
                    'total_normal_force', obj.filled('right_wheel_total_normal_force'), ...
                    'max_normal_force', obj.filled('right_wheel_max_normal_force') ...
                ), ...
                'surface', struct( ...
                    'count', obj.filled('surface_contact_count'), ...
                    'total_force_magnitude', obj.filled('surface_total_force_magnitude'), ...
                    'total_normal_force', obj.filled('surface_total_normal_force'), ...
                    'max_normal_force', obj.filled('surface_max_normal_force') ...
                ), ...
                'wall', struct( ...
                    'count', obj.filled('wall_contact_count'), ...
                    'total_force_magnitude', obj.filled('wall_total_force_magnitude'), ...
                    'total_normal_force', obj.filled('wall_total_normal_force'), ...
                    'max_normal_force', obj.filled('wall_max_normal_force') ...
                ), ...
                'details', {obj.filled('contact_details')} ...
            );
            % Caller-registered extension channels (e.g. a manual controller's
            % command telemetry) are written as their own top-level group.
            if ~isempty(obj.extension.group_name)
                extension_group = struct();
                for channel_index = 1:size(obj.extension.channels, 1)
                    channel_name = obj.extension.channels{channel_index, 1};
                    extension_group.(channel_name) = obj.filled([obj.extension.group_name '_' channel_name]);
                end
                log.(obj.extension.group_name) = extension_group;
            end

            save(obj.file_path, 'log');
        end

        function ensure_capacity(obj, required_capacity)
            if required_capacity <= obj.capacity
                return;
            end

            new_capacity = max(required_capacity, obj.capacity * 2);
            for channel_index = 1:numel(obj.schema)
                channel = obj.schema(channel_index);
                grown_buffer = obj.make_buffer(channel.width, channel.kind, new_capacity);
                if iscell(grown_buffer)
                    grown_buffer(1:obj.count, 1) = obj.buffers.(channel.name)(1:obj.count, 1);
                else
                    grown_buffer(1:obj.count, :) = obj.buffers.(channel.name)(1:obj.count, :);
                end
                obj.buffers.(channel.name) = grown_buffer;
            end
            obj.capacity = new_capacity;
        end
    end
end


function extension = normalize_extension(extension_input)
% Validate/normalize the constructor's extension definition. Inactive when no
% group is given: group_name '' and an empty channel list.
extension = struct('group_name', '', 'channels', {cell(0, 2)});
if ~isstruct(extension_input) || ~isfield(extension_input, 'group_name')
    return;
end
group_name = char(extension_input.group_name);
if isempty(group_name)
    return;
end
if ~isvarname(group_name)
    error('uavsim:badLogExtension', 'extension.group_name must be a valid identifier (got "%s").', group_name);
end
reserved_groups = {'meta', 'config', 'options', 'state', 'sensor_truth', 'control', 'actuator', 'network', 'reference', 'contact'};
if any(strcmp(group_name, reserved_groups))
    error('uavsim:badLogExtension', 'extension.group_name "%s" collides with a core log group.', group_name);
end
if ~isfield(extension_input, 'channels') || ~iscell(extension_input.channels) || size(extension_input.channels, 2) ~= 2
    error('uavsim:badLogExtension', 'extension.channels must be an N x 2 cell {name, width; ...}.');
end
for channel_index = 1:size(extension_input.channels, 1)
    channel_name = extension_input.channels{channel_index, 1};
    channel_width = extension_input.channels{channel_index, 2};
    if ~isvarname(channel_name)
        error('uavsim:badLogExtension', 'extension channel name "%s" must be a valid identifier.', char(channel_name));
    end
    if ~(isnumeric(channel_width) && isscalar(channel_width) && channel_width >= 1 && channel_width == floor(channel_width))
        error('uavsim:badLogExtension', 'extension channel "%s" width must be a positive integer.', channel_name);
    end
end
extension.group_name = group_name;
extension.channels = extension_input.channels;
end


function schema = append_extension_schema(schema, extension)
% Extension channels become ordinary NaN-filled numeric buffers named
% <group_name>_<channel_name>.
for channel_index = 1:size(extension.channels, 1)
    schema(end + 1) = struct( ...
        'name', [extension.group_name '_' extension.channels{channel_index, 1}], ...
        'width', extension.channels{channel_index, 2}, ...
        'kind', 'nan' ...
    ); %#ok<AGROW>
end
end


function row = add_extension_row(row, extension, extension_status)
% Fill the registered extension channels from the status struct; missing or
% mis-sized fields stay NaN so a partial status never errors mid-run.
if isempty(extension.group_name)
    return;
end
if ~isstruct(extension_status)
    extension_status = struct();
end
for channel_index = 1:size(extension.channels, 1)
    channel_name = extension.channels{channel_index, 1};
    channel_width = extension.channels{channel_index, 2};
    value = nan(1, channel_width);
    if isfield(extension_status, channel_name) && ~isempty(extension_status.(channel_name))
        raw_value = reshape(double(extension_status.(channel_name)), 1, []);
        if channel_width == 1
            value = raw_value(1);
        elseif numel(raw_value) == channel_width
            value = raw_value;
        end
    end
    row.([extension.group_name '_' channel_name]) = value;
end
end


function row = build_sample_row(state, control_command, target_position)
% One appended sample as a struct keyed by channel_schema() names.
[logged_rotor_thrusts, logged_rotor_omega] = unpack_control_command(control_command);
contact_payload = extract_contact_payload(state);
state_network_payload = extract_state_network_payload(state);
command_network_payload = extract_command_network_payload(control_command);
sensor_truth_payload = extract_sensor_truth_payload(state);
actuator_payload = extract_actuator_payload(state);

row = struct();
row.time = double(state.time);
row.realtime_factor = get_realtime_factor(state);
row.state_protocol_version = state_network_payload.protocol_version;
row.state_sequence = state_network_payload.sequence;
row.state_wall_time_send_ns = state_network_payload.wall_time_send_ns;
row.state_age_ms = state_network_payload.age_ms;
row.state_fidelity_mode = state_network_payload.fidelity_mode;
row.position = reshape(double(state.position), 1, 3);
row.velocity = reshape(double(state.velocity), 1, 3);
row.angular_velocity_body = reshape(double(state.angular_velocity_body), 1, 3);
row.angular_velocity_world = reshape(double(state.angular_velocity_world), 1, 3);
row.rotation_matrix = reshape(double(state.rotation_matrix), 1, 9);
row.sensor_truth_position = sensor_truth_payload.position;
row.sensor_truth_velocity = sensor_truth_payload.velocity;
row.sensor_truth_angular_velocity_body = sensor_truth_payload.angular_velocity_body;
row.sensor_truth_angular_velocity_world = sensor_truth_payload.angular_velocity_world;
row.sensor_truth_rotation_matrix = sensor_truth_payload.rotation_matrix;
row.rotor_thrusts = reshape(double(logged_rotor_thrusts), 1, 4);
row.rotor_omega = reshape(double(logged_rotor_omega), 1, 4);
row.command_protocol_version = command_network_payload.protocol_version;
row.command_sequence = command_network_payload.sequence;
row.command_source_state_sequence = command_network_payload.source_state_sequence;
row.command_wall_time_send_ns = command_network_payload.wall_time_send_ns;
row.command_fidelity_mode = command_network_payload.fidelity_mode;
row.controller_compute_ms = command_network_payload.controller_compute_ms;
row.state_sequence_gap = command_network_payload.state_sequence_gap;
row.actuator_requested_rotor_thrusts = actuator_payload.requested_rotor_thrusts;
row.actuator_applied_rotor_thrusts = actuator_payload.applied_rotor_thrusts;
row.actuator_tracking_error = actuator_payload.tracking_error;
row.target_position = reshape(double(target_position), 1, 3);
row.contact_count = contact_payload.count;
row.total_contact_force_magnitude = contact_payload.total_force_magnitude;
row.max_contact_force_magnitude = contact_payload.max_force_magnitude;
row.total_contact_normal_force = contact_payload.total_normal_force;
row.max_contact_normal_force = contact_payload.max_normal_force;
row.left_wheel_contact_count = contact_payload.left_wheel.count;
row.left_wheel_total_force_magnitude = contact_payload.left_wheel.total_force_magnitude;
row.left_wheel_total_normal_force = contact_payload.left_wheel.total_normal_force;
row.left_wheel_max_normal_force = contact_payload.left_wheel.max_normal_force;
row.right_wheel_contact_count = contact_payload.right_wheel.count;
row.right_wheel_total_force_magnitude = contact_payload.right_wheel.total_force_magnitude;
row.right_wheel_total_normal_force = contact_payload.right_wheel.total_normal_force;
row.right_wheel_max_normal_force = contact_payload.right_wheel.max_normal_force;
row.surface_contact_count = contact_payload.surface.count;
row.surface_total_force_magnitude = contact_payload.surface.total_force_magnitude;
row.surface_total_normal_force = contact_payload.surface.total_normal_force;
row.surface_max_normal_force = contact_payload.surface.max_normal_force;
row.wall_contact_count = contact_payload.wall.count;
row.wall_total_force_magnitude = contact_payload.wall.total_force_magnitude;
row.wall_total_normal_force = contact_payload.wall.total_normal_force;
row.wall_max_normal_force = contact_payload.wall.max_normal_force;
row.contact_details = contact_payload.details;
end


% Thin delegations to the shared protocol helpers (single source of truth:
% when a contact group is added, only uavsim.Protocol needs the change).
function value = get_contact_summary_field(state, field_name)
value = uavsim.Protocol.get_contact_summary_field(state, field_name);
end


function value = get_realtime_factor(state)
value = uavsim.Protocol.get_realtime_factor(state);
end


function value = get_nested_contact_summary_field(state, group_name, field_name)
value = uavsim.Protocol.get_contact_group_field(state, group_name, field_name);
end


function contact_payload = extract_contact_payload(state)
% 'details' is wrapped in {} so struct() stores it as ONE value: a cell (or
% struct-array) value would otherwise fan the whole payload out into a
% struct array and break the field accesses downstream.
contact_payload = struct( ...
    'count', get_contact_summary_field(state, 'count'), ...
    'total_force_magnitude', get_contact_summary_field(state, 'total_force_magnitude'), ...
    'max_force_magnitude', get_contact_summary_field(state, 'max_force_magnitude'), ...
    'total_normal_force', get_contact_summary_field(state, 'total_normal_force'), ...
    'max_normal_force', get_contact_summary_field(state, 'max_normal_force'), ...
    'left_wheel', build_contact_group_payload(state, 'left_wheel'), ...
    'right_wheel', build_contact_group_payload(state, 'right_wheel'), ...
    'surface', build_contact_group_payload(state, 'surface'), ...
    'wall', build_contact_group_payload(state, 'wall'), ...
    'details', {get_contact_details(state)} ...
);
end


function group_payload = build_contact_group_payload(state, group_name)
group_payload = struct( ...
    'count', get_nested_contact_summary_field(state, group_name, 'count'), ...
    'total_force_magnitude', get_nested_contact_summary_field(state, group_name, 'total_force_magnitude'), ...
    'total_normal_force', get_nested_contact_summary_field(state, group_name, 'total_normal_force'), ...
    'max_normal_force', get_nested_contact_summary_field(state, group_name, 'max_normal_force') ...
);
end


function details = get_contact_details(state)
details = struct([]);
if ~isfield(state, 'contacts')
    return;
end

details = state.contacts;
if iscell(details)
    % jsondecode returns a CELL array when the per-contact JSON objects carry
    % different field sets — surface contacts include surface_name /
    % surface_height / surface_normal while wall contacts do not, so touching
    % floor and wall in the same sample mixes the shapes. Fill the missing
    % fields so every sample stores a uniform struct array, which is what the
    % review scripts expect.
    details = normalize_heterogeneous_contacts(details);
end
end


function details = normalize_heterogeneous_contacts(raw_contacts)
if isempty(raw_contacts)
    details = struct([]);
    return;
end

all_field_names = {};
for contact_index = 1:numel(raw_contacts)
    all_field_names = union(all_field_names, fieldnames(raw_contacts{contact_index}), 'stable');
end

for contact_index = 1:numel(raw_contacts)
    contact_entry = raw_contacts{contact_index};
    for field_index = 1:numel(all_field_names)
        field_name = all_field_names{field_index};
        if ~isfield(contact_entry, field_name)
            contact_entry.(field_name) = [];
        end
    end
    raw_contacts{contact_index} = orderfields(contact_entry, all_field_names);
end

details = reshape([raw_contacts{:}], [], 1);
end


function [rotor_thrusts, rotor_omega] = unpack_control_command(control_command)
rotor_thrusts = nan(1, 4);
rotor_omega = nan(1, 4);

if isnumeric(control_command)
    rotor_thrusts = reshape(double(control_command), 1, 4);
    return;
end

if ~isstruct(control_command)
    error('uavsim:badControlCommand', 'control_command must be numeric or struct.');
end

if isfield(control_command, 'rotor_thrusts')
    rotor_thrusts = reshape(double(control_command.rotor_thrusts), 1, 4);
end

if isfield(control_command, 'rotor_omega')
    rotor_omega = reshape(double(control_command.rotor_omega), 1, 4);
end
end


function payload = extract_state_network_payload(state)
payload = struct( ...
    'protocol_version', 1.0, ...
    'sequence', NaN, ...
    'wall_time_send_ns', NaN, ...
    'age_ms', NaN, ...
    'fidelity_mode', '' ...
);

if ~isfield(state, 'packet_metrics')
    return;
end

metrics = state.packet_metrics;
payload.protocol_version = get_struct_numeric(metrics, 'protocol_version', 1.0);
payload.sequence = get_struct_numeric(metrics, 'sequence', NaN);
payload.wall_time_send_ns = get_struct_numeric(metrics, 'wall_time_send_ns', NaN);
payload.age_ms = get_struct_numeric(metrics, 'age_ms', NaN);
payload.fidelity_mode = get_struct_char(metrics, 'fidelity_mode', '');
end


function payload = extract_command_network_payload(control_command)
payload = struct( ...
    'protocol_version', 1.0, ...
    'sequence', NaN, ...
    'source_state_sequence', NaN, ...
    'wall_time_send_ns', NaN, ...
    'fidelity_mode', '', ...
    'controller_compute_ms', NaN, ...
    'state_sequence_gap', NaN ...
);

if isfield(control_command, 'packet_metadata')
    metadata = control_command.packet_metadata;
    payload.protocol_version = get_struct_numeric(metadata, 'protocol_version', 1.0);
    payload.sequence = get_struct_numeric(metadata, 'sequence', NaN);
    payload.source_state_sequence = get_struct_numeric(metadata, 'source_state_sequence', NaN);
    payload.wall_time_send_ns = get_struct_numeric(metadata, 'wall_time_send_ns', NaN);
    payload.fidelity_mode = get_struct_char(metadata, 'fidelity_mode', '');
end

if isfield(control_command, 'runtime_metrics')
    metrics = control_command.runtime_metrics;
    payload.controller_compute_ms = get_struct_numeric(metrics, 'controller_compute_ms', NaN);
    payload.state_sequence_gap = get_struct_numeric(metrics, 'state_sequence_gap', NaN);
end
end


function value = get_struct_numeric(input_struct, field_name, default_value)
value = default_value;
if ~isstruct(input_struct) || ~isfield(input_struct, field_name)
    return;
end

raw_value = input_struct.(field_name);
if isempty(raw_value)
    return;
end

value = double(raw_value);
end


function value = get_struct_char(input_struct, field_name, default_value)
value = default_value;
if ~isstruct(input_struct) || ~isfield(input_struct, field_name)
    return;
end

raw_value = input_struct.(field_name);
if ischar(raw_value)
    value = raw_value;
elseif isstring(raw_value) && isscalar(raw_value)
    value = char(raw_value);
end
end


function payload = extract_sensor_truth_payload(state)
payload = struct( ...
    'position', nan(1, 3), ...
    'velocity', nan(1, 3), ...
    'angular_velocity_body', nan(1, 3), ...
    'angular_velocity_world', nan(1, 3), ...
    'rotation_matrix', nan(1, 9) ...
);

if ~isfield(state, 'sensor_truth')
    return;
end

truth = state.sensor_truth;
payload.position = get_struct_row_vector(truth, 'position', 3);
payload.velocity = get_struct_row_vector(truth, 'velocity', 3);
payload.angular_velocity_body = get_struct_row_vector(truth, 'angular_velocity_body', 3);
payload.angular_velocity_world = get_struct_row_vector(truth, 'angular_velocity_world', 3);
payload.rotation_matrix = get_struct_row_vector(truth, 'rotation_matrix', 9);
end


function payload = extract_actuator_payload(state)
payload = struct( ...
    'requested_rotor_thrusts', nan(1, 4), ...
    'applied_rotor_thrusts', nan(1, 4), ...
    'tracking_error', nan(1, 4) ...
);

if ~isfield(state, 'actuator')
    return;
end

actuator = state.actuator;
payload.requested_rotor_thrusts = get_struct_row_vector(actuator, 'requested_rotor_thrusts', 4);
payload.applied_rotor_thrusts = get_struct_row_vector(actuator, 'applied_rotor_thrusts', 4);
payload.tracking_error = get_struct_row_vector(actuator, 'tracking_error', 4);
end


function value = get_struct_row_vector(input_struct, field_name, expected_length)
value = nan(1, expected_length);
if ~isstruct(input_struct) || ~isfield(input_struct, field_name)
    return;
end

raw_value = reshape(double(input_struct.(field_name)), 1, []);
if numel(raw_value) ~= expected_length
    return;
end

value = raw_value;
end
