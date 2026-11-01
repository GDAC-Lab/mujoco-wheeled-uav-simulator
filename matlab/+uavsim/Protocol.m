classdef Protocol
    % UAVSIM.PROTOCOL  UDP wire format: state packets in, command packets out.
    %
    % Mirrors wheeled_uav/protocol.py. Timing contract (docs/TIMING.md):
    %   - Control uses state.time only.
    %   - One evaluation per new UDP sample (udp_state_is_new).
    %   - build_sync_metrics is for logging; never pause on sim_wall_skew.
    methods (Static)
        function state = read_latest_state(controller_socket)
            % Drain the receive queue and decode the newest datagram (or []).
            if controller_socket.NumDatagramsAvailable == 0
                state = [];
                return;
            end

            data_packet = read(controller_socket, controller_socket.NumDatagramsAvailable, "string");
            latest_packet = data_packet(end).Data;
            state = jsondecode(latest_packet);
            state = uavsim.Protocol.attach_state_packet_metrics(state);
        end

        function state = attach_state_packet_metrics(state)
            if ~isstruct(state)
                return;
            end

            packet_metrics = uavsim.Protocol.extract_packet_metrics(state);
            if isfield(state, 'uavs')
                % jsondecode yields a CELL array when the per-UAV objects have
                % different field sets (e.g. differing contact payloads), so
                % handle both shapes here instead of assuming a struct array.
                if iscell(state.uavs)
                    for uav_index = 1:numel(state.uavs)
                        state.uavs{uav_index}.packet_metrics = packet_metrics;
                    end
                else
                    for uav_index = 1:numel(state.uavs)
                        state.uavs(uav_index).packet_metrics = packet_metrics;
                    end
                end
                return;
            end
            state.packet_metrics = packet_metrics;
        end

        function uav_states = uav_state_list(state)
            % Normalize state.uavs to a numel x 1 cell of scalar structs,
            % regardless of whether jsondecode produced a struct array or a
            % cell array (heterogeneous per-UAV fields flip it to a cell).
            if ~isstruct(state) || ~isfield(state, 'uavs')
                uav_states = {};
                return;
            end
            raw_uav_states = state.uavs;
            if iscell(raw_uav_states)
                uav_states = raw_uav_states(:);
            else
                uav_states = num2cell(raw_uav_states(:));
            end
        end

        function assert_single_uav_state(state, controller_name)
            % Single-UAV controllers paired with `simulate --num-uavs N` fail
            % much later with a cryptic missing-field error; fail fast with the
            % same wording as the Python reference controller instead.
            if isstruct(state) && isfield(state, 'uavs')
                error('uavsim:multiUavPacket', ...
                    '%s expects single-UAV state packets, but received a multi-UAV packet (run the simulator without --num-uavs, or use multi_uav_formation_controller).', ...
                    controller_name);
            end
        end

        function state_metrics = get_state_packet_metrics(state)
            if isstruct(state) && isfield(state, 'packet_metrics')
                metrics = state.packet_metrics;
                if isstruct(metrics) && isscalar(metrics) && isequal(fieldnames(metrics), ...
                        {'protocol_version'; 'sequence'; 'wall_time_send_ns'; 'fidelity_mode'; 'age_ms'})
                    % Already normalized by read_latest_state (extract_packet_metrics):
                    % hand it back instead of rebuilding it field by field, since
                    % controllers call this once per UAV and control period.
                    state_metrics = metrics;
                    return;
                end
            end
            default_metrics = struct( ...
                'protocol_version', 1.0, ...
                'sequence', NaN, ...
                'wall_time_send_ns', NaN, ...
                'fidelity_mode', '', ...
                'age_ms', NaN ...
            );
            if ~isstruct(state) || ~isfield(state, 'packet_metrics')
                state_metrics = default_metrics;
                return;
            end

            metrics = state.packet_metrics;
            state_metrics = struct( ...
                'protocol_version', uavsim.Util.get_struct_field(metrics, 'protocol_version', 1.0), ...
                'sequence', uavsim.Util.default_numeric(uavsim.Util.get_struct_field(metrics, 'sequence', NaN)), ...
                'wall_time_send_ns', uavsim.Util.default_numeric(uavsim.Util.get_struct_field(metrics, 'wall_time_send_ns', NaN)), ...
                'fidelity_mode', char(uavsim.Util.get_struct_field(metrics, 'fidelity_mode', '')), ...
                'age_ms', uavsim.Util.default_numeric(uavsim.Util.get_struct_field(metrics, 'age_ms', NaN)) ...
            );
        end

        function ts = sim_time_seconds(state)
            % Simulator JSON state.time (seconds), or NaN if missing.
            ts = NaN;
            if isempty(state) || ~isstruct(state) || ~isfield(state, 'time')
                return;
            end
            raw = state.time;
            if isnumeric(raw) && ~isempty(raw)
                ts = double(raw(1));
            end
        end

        function key = udp_state_sample_key(state)
            % Composite key over (protocol sequence, simulator time).
            t_s = uavsim.Protocol.sim_time_seconds(state);
            seq = NaN;
            if isstruct(state) && isfield(state, 'packet_metrics')
                pm = state.packet_metrics;
                if isfield(pm, 'sequence') && ~isempty(pm.sequence)
                    seq = double(pm.sequence);
                end
            elseif isstruct(state) && isfield(state, 'sequence') && ~isempty(state.sequence)
                seq = double(state.sequence);
            end
            if ~isnan(seq)
                if ~isnan(t_s)
                    key = sprintf('seq=%.0f|t=%.17g', seq, t_s);
                else
                    key = sprintf('seq=%.0f', seq);
                end
            else
                key = sprintf('t=%.17g', t_s);
            end
        end

        function [is_new, tracker] = udp_state_is_new(state, tracker_in)
            % True when this state is a new simulator sample (drops duplicate UDP reads).
            if nargin < 2 || isempty(tracker_in)
                tracker_in = struct('last_key', '');
            end
            key = uavsim.Protocol.udp_state_sample_key(state);
            is_new = isempty(tracker_in.last_key) || ~strcmp(key, tracker_in.last_key);
            tracker = tracker_in;
            if is_new
                tracker.last_key = key;
            end
        end

        function sync = build_sync_metrics(state)
            % Diagnostic timing metrics from the simulator state packet.
            % Controllers must NOT wait on sim_wall_skew — logging only.
            sync = struct( ...
                'sim_time_seconds', uavsim.Protocol.sim_time_seconds(state), ...
                'realtime_factor', NaN, ...
                'packet_age_ms', NaN, ...
                'control_period_seconds', NaN, ...
                'sim_wall_skew_seconds', NaN, ...
                'session_wall_elapsed_seconds', NaN ...
            );
            if isempty(state) || ~isstruct(state)
                return;
            end
            if isfield(state, 'realtime_factor')
                sync.realtime_factor = double(state.realtime_factor);
            end
            if isfield(state, 'packet_metrics') && isstruct(state.packet_metrics)
                pm = state.packet_metrics;
                if isfield(pm, 'age_ms')
                    sync.packet_age_ms = double(pm.age_ms);
                end
            end
            if isfield(state, 'timing') && isstruct(state.timing)
                timing = state.timing;
                if isfield(timing, 'control_period_seconds')
                    sync.control_period_seconds = double(timing.control_period_seconds);
                end
                if isfield(timing, 'sim_wall_skew_seconds')
                    sync.sim_wall_skew_seconds = double(timing.sim_wall_skew_seconds);
                end
                if isfield(timing, 'session_wall_elapsed_seconds')
                    sync.session_wall_elapsed_seconds = double(timing.session_wall_elapsed_seconds);
                end
                if isfield(timing, 'realtime_factor') && isnan(sync.realtime_factor)
                    sync.realtime_factor = double(timing.realtime_factor);
                end
            end
        end

        function value = get_contact_summary_field(state, field_name)
            value = 0.0;
            if ~isfield(state, 'contact_summary')
                return;
            end
            if ~isfield(state.contact_summary, field_name)
                return;
            end

            value = double(state.contact_summary.(field_name));
        end

        function value = get_contact_group_field(state, group_name, field_name)
            % Per-group contact summary value (left_wheel / right_wheel /
            % surface / wall); 0.0 when the packet does not carry the group
            % (e.g. logs or simulators from before the wall group existed).
            value = 0.0;
            if ~isfield(state, 'contact_summary') || ~isfield(state.contact_summary, group_name)
                return;
            end
            group = state.contact_summary.(group_name);
            if ~isfield(group, field_name)
                return;
            end
            value = double(group.(field_name));
        end

        function value = get_realtime_factor(state)
            value = 0.0;
            if ~isfield(state, 'realtime_factor')
                return;
            end

            value = double(state.realtime_factor);
        end

        % -------------------------------------------------------------------
        % Outgoing command packets
        % -------------------------------------------------------------------

        function command_options = build_command_options(command_mode, thrust_coefficient, varargin)
            parser = inputParser;
            addParameter(parser, 'fidelity_mode', 'baseline', @(value) ischar(value) || (isstring(value) && isscalar(value)));
            parse(parser, varargin{:});

            uavsim.Protocol.validate_command_mode(command_mode);
            command_options = struct( ...
                'input_mode', command_mode, ...
                'thrust_coefficient', thrust_coefficient, ...
                'fidelity_mode', char(parser.Results.fidelity_mode) ...
            );
        end

        function control_command = build_control_command(rotor_thrusts, command_options, varargin)
            % Name-value options: sequence, source_state_sequence,
            % wall_time_send_ns, controller_compute_ms, state_age_ms,
            % state_sequence_gap (numeric scalars or []), fidelity_mode (text)
            % and body_wrench (6 numbers or []). Names match as with
            % inputParser (case-insensitive, unique prefixes, an options struct
            % expands) but are parsed by hand: this runs once per UAV and
            % control period, and an inputParser costs about 0.3 ms per call.
            options = uavsim.Protocol.parse_command_options(command_options.fidelity_mode, varargin);

            control_command = struct('rotor_thrusts', rotor_thrusts(:)');
            if ~isempty(options.body_wrench)
                % Optional world-frame external wrench [Fx,Fy,Fz,Mx,My,Mz]
                % (tilt-rotor emulation), applied via data.xfrc_applied.
                control_command.body_wrench = reshape(double(options.body_wrench), 1, 6);
            end
            if strcmp(command_options.input_mode, 'omega')
                control_command.rotor_omega = uavsim.Protocol.thrust_to_rotor_omega(rotor_thrusts, command_options.thrust_coefficient)';
            end

            if ~isempty(options.sequence)
                control_command.packet_metadata = struct( ...
                    'protocol_version', 2, ...
                    'sequence', double(options.sequence), ...
                    'source_state_sequence', double(options.source_state_sequence), ...
                    'wall_time_send_ns', double(options.wall_time_send_ns), ...
                    'fidelity_mode', char(options.fidelity_mode) ...
                );
            end
            control_command.runtime_metrics = struct( ...
                'controller_compute_ms', uavsim.Util.default_numeric(options.controller_compute_ms), ...
                'state_age_ms', uavsim.Util.default_numeric(options.state_age_ms), ...
                'state_sequence_gap', uavsim.Util.default_numeric(options.state_sequence_gap) ...
            );
        end

        function send_control_command(controller_socket, control_command, target_ip, target_port)
            message = struct();
            if isfield(control_command, 'packet_metadata')
                metadata = control_command.packet_metadata;
                message.protocol_version = double(metadata.protocol_version);
                message.sequence = double(metadata.sequence);
                message.source_state_sequence = double(metadata.source_state_sequence);
                message.wall_time_send_ns = double(metadata.wall_time_send_ns);
                message.fidelity_mode = char(metadata.fidelity_mode);
            end
            if strcmp(uavsim.Protocol.control_command_mode(control_command), 'omega')
                message.rotor_omega = control_command.rotor_omega;
            else
                message.rotor_thrusts = control_command.rotor_thrusts;
            end
            if isfield(control_command, 'body_wrench')
                message.body_wrench = control_command.body_wrench;
            end
            write(controller_socket, jsonencode(message), "string", target_ip, target_port);
        end

        function send_multi_uav_control_command(controller_socket, control_commands, command_options, target_ip, target_port)
            num_uavs = numel(control_commands);
            command_matrix = zeros(num_uavs, 4);
            for uav_index = 1:num_uavs
                command_values = uavsim.Protocol.displayed_command_values(control_commands{uav_index}, command_options);
                command_matrix(uav_index, :) = reshape(double(command_values), 1, 4);
            end

            message = struct();
            if num_uavs >= 1 && isstruct(control_commands{1}) && isfield(control_commands{1}, 'packet_metadata')
                metadata = control_commands{1}.packet_metadata;
                message.protocol_version = double(metadata.protocol_version);
                message.sequence = double(metadata.sequence);
                message.source_state_sequence = double(metadata.source_state_sequence);
                message.wall_time_send_ns = double(metadata.wall_time_send_ns);
                message.fidelity_mode = char(metadata.fidelity_mode);
            end
            if strcmp(command_options.input_mode, 'omega')
                message.rotor_omegas = command_matrix;
            else
                message.rotor_thrusts = command_matrix;
            end
            % Optional per-UAV world-frame body wrench (tilt-rotor emulation).
            body_wrenches = zeros(num_uavs, 6);
            has_wrench = false;
            for uav_index = 1:num_uavs
                if isstruct(control_commands{uav_index}) && isfield(control_commands{uav_index}, 'body_wrench')
                    body_wrenches(uav_index, :) = reshape(double(control_commands{uav_index}.body_wrench), 1, 6);
                    has_wrench = true;
                end
            end
            if has_wrench
                message.body_wrenches = body_wrenches;
            end
            write(controller_socket, jsonencode(message), "string", target_ip, target_port);
        end

        function values = displayed_command_values(control_command, command_options)
            if strcmp(command_options.input_mode, 'omega')
                values = reshape(double(control_command.rotor_omega), [], 1);
                return;
            end

            values = reshape(double(control_command.rotor_thrusts), [], 1);
        end

        function unit_label = command_unit_label(input_mode)
            if strcmp(input_mode, 'omega')
                unit_label = 'rad/s';
                return;
            end

            unit_label = 'N';
        end

        function rotor_omega = thrust_to_rotor_omega(rotor_thrusts, thrust_coefficient)
            if thrust_coefficient <= 0.0
                error('uavsim:invalidThrustCoefficient', 'thrust_coefficient must be positive when input_mode is omega.');
            end

            rotor_omega = sqrt(max(0.0, rotor_thrusts) ./ thrust_coefficient);
        end
    end

    methods (Static, Access = private)
        function options = parse_command_options(default_fidelity_mode, args)
            options = struct( ...
                'sequence', [], ...
                'source_state_sequence', [], ...
                'wall_time_send_ns', [], ...
                'controller_compute_ms', [], ...
                'state_age_ms', [], ...
                'state_sequence_gap', [], ...
                'fidelity_mode', default_fidelity_mode, ...
                'body_wrench', [] ...
            );
            if isscalar(args) && isstruct(args{1})
                args = reshape([fieldnames(args{1})'; struct2cell(args{1})'], 1, []);
            end
            if mod(numel(args), 2) ~= 0
                error('uavsim:Protocol:optionWithoutValue', 'Options must be given as name-value pairs.');
            end
            option_names = fieldnames(options);
            for index = 1:2:numel(args)
                name = args{index};
                if ~(ischar(name) || (isstring(name) && isscalar(name)))
                    error('uavsim:Protocol:invalidOptionName', 'Option names must be text.');
                end
                name = char(name);
                matches = strcmpi(option_names, name);
                if ~any(matches)
                    matches = strncmpi(option_names, name, numel(name));
                end
                if nnz(matches) ~= 1
                    error('uavsim:Protocol:unknownOption', '''%s'' is not a recognized or unique option.', name);
                end
                option_name = option_names{matches};
                value = args{index + 1};
                switch option_name
                    case 'fidelity_mode'
                        is_valid = ischar(value) || (isstring(value) && isscalar(value));
                    case 'body_wrench'
                        is_valid = isempty(value) || (isnumeric(value) && numel(value) == 6);
                    otherwise
                        is_valid = isempty(value) || (isnumeric(value) && isscalar(value));
                end
                if ~is_valid
                    error('uavsim:Protocol:invalidOptionValue', 'The value of ''%s'' is invalid.', option_name);
                end
                options.(option_name) = value;
            end
        end

        function packet_metrics = extract_packet_metrics(packet_struct)
            wall_time_send_ns = uavsim.Util.get_struct_field(packet_struct, 'wall_time_send_ns', NaN);
            receive_time_ns = uavsim.Util.wall_time_now_ns();
            age_ms = NaN;
            if ~isempty(wall_time_send_ns) && ~isnan(double(wall_time_send_ns))
                age_ms = max(0.0, (double(receive_time_ns) - double(wall_time_send_ns)) / 1.0e6);
            end

            packet_metrics = struct( ...
                'protocol_version', double(uavsim.Util.get_struct_field(packet_struct, 'protocol_version', 1.0)), ...
                'sequence', uavsim.Util.default_numeric(uavsim.Util.get_struct_field(packet_struct, 'sequence', NaN)), ...
                'wall_time_send_ns', uavsim.Util.default_numeric(wall_time_send_ns), ...
                'fidelity_mode', char(uavsim.Util.get_struct_field(packet_struct, 'fidelity_mode', '')), ...
                'age_ms', uavsim.Util.default_numeric(age_ms) ...
            );
        end

        function mode_name = control_command_mode(control_command)
            if isfield(control_command, 'rotor_omega')
                mode_name = 'omega';
                return;
            end

            mode_name = 'thrust';
        end

        function validate_command_mode(command_mode)
            if strcmp(command_mode, 'thrust') || strcmp(command_mode, 'omega')
                return;
            end

            error('uavsim:invalidCommandMode', 'Unsupported command_mode: %s', command_mode);
        end
    end
end
