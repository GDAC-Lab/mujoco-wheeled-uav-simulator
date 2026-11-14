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

            data_packet = read(controller_socket, controller_socket.NumDatagramsAvailable, "uint8");
            state = uavsim.Protocol.decode_state(data_packet(end).Data);
            state = uavsim.Protocol.attach_state_packet_metrics(state);
        end

        function state = decode_state(datagram)
            % Decode a state datagram, JSON or binary (simulation.state_format;
            % see wheeled_uav/state_packets.py), into the struct jsondecode
            % gives for the JSON packet of the same state.
            datagram = reshape(uint8(datagram), 1, []);
            if numel(datagram) >= 4 && all(datagram(1:4) == uint8('WUAV'))
                state = uavsim.Protocol.decode_binary_state(datagram);
            else
                state = jsondecode(native2unicode(datagram, 'UTF-8'));
            end
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
            % state_sequence_gap (numeric scalars or []), fidelity_mode (text),
            % body_wrench (6 numbers or []) and extensions (a struct of
            % extension name -> command, e.g. struct('goal', struct('position',
            % [1 0 1])), or []; docs/EXTENSIONS.md). Names match as with
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
            if ~isempty(options.extensions)
                control_command.extensions = options.extensions;
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
            write(controller_socket, uavsim.Protocol.encode_control_command(control_command), "string", target_ip, target_port);
        end

        function text = encode_control_command(control_command)
            % The JSON text of the command packet send_control_command sends.
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
            if isfield(control_command, 'extensions')
                message.extensions = control_command.extensions;
            end
            text = jsonencode(message);
        end

        function send_multi_uav_control_command(controller_socket, control_commands, command_options, target_ip, target_port)
            text = uavsim.Protocol.encode_multi_uav_control_command(control_commands, command_options);
            write(controller_socket, text, "string", target_ip, target_port);
        end

        function text = encode_multi_uav_control_command(control_commands, command_options)
            % The JSON text of the command packet send_multi_uav_control_command sends.
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
            % Extension commands belong to the scene, not to one UAV: the
            % extensions of every per-UAV command go into one object (for a
            % name given twice, the later UAV's command is sent).
            extensions = struct();
            has_extensions = false;
            for uav_index = 1:num_uavs
                if isstruct(control_commands{uav_index}) && isfield(control_commands{uav_index}, 'extensions')
                    given = control_commands{uav_index}.extensions;
                    names = fieldnames(given);
                    for name_index = 1:numel(names)
                        extensions.(names{name_index}) = given.(names{name_index});
                    end
                    has_extensions = true;
                end
            end
            if has_extensions
                message.extensions = extensions;
            end
            text = jsonencode(message);
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
        function state = decode_binary_state(datagram)
            % Layout version 1 of wheeled_uav/state_packets.py: a 52-byte header,
            % ten timing values, the UAV names, then one float64 row per UAV.
            if typecast(datagram(5:6), 'uint16') ~= 1
                error('uavsim:Protocol:binaryLayout', 'Unsupported binary state packet layout %d.', ...
                    typecast(datagram(5:6), 'uint16'));
            end
            num_uavs = double(typecast(datagram(9:10), 'uint16'));
            record_length = double(typecast(datagram(11:12), 'uint16'));
            flags = double(typecast(datagram(15:16), 'uint16'));
            instance_id = double(typecast(datagram(17:20), 'int32'));
            sim_time_and_rtf = typecast(datagram(37:52), 'double');
            timing_values = typecast(datagram(53:132), 'double');
            names_length = double(typecast(datagram(133:134), 'uint16'));
            name_bytes = datagram(135:134 + names_length);
            if all(name_bytes < 128)
                name_text = char(name_bytes);
            else
                name_text = native2unicode(name_bytes, 'UTF-8');
            end
            breaks = [0, find(name_text == char(10)), numel(name_text) + 1];
            names = cell(1, num_uavs);
            for uav_index = 1:num_uavs
                names{uav_index} = name_text(breaks(uav_index) + 1:breaks(uav_index + 1) - 1);
            end
            offset = 134 + names_length;
            offset = offset + mod(-offset, 8);
            values = reshape(typecast(datagram(offset + 1:offset + 8 * record_length * num_uavs), 'double'), ...
                record_length, num_uavs);

            fidelity_modes = {'baseline', 'hil'};
            fidelity_mode = [];   % code 255: no mode (jsondecode gives [] for null)
            if datagram(13) < numel(fidelity_modes)
                fidelity_mode = fidelity_modes{double(datagram(13)) + 1};
            end
            state = struct( ...
                'protocol_version', double(typecast(datagram(7:8), 'uint16')), ...
                'sequence', double(typecast(datagram(21:28), 'int64')), ...
                'wall_time_send_ns', double(typecast(datagram(29:36), 'int64')), ...
                'fidelity_mode', fidelity_mode, ...
                'sim_time', sim_time_and_rtf(1) ...
            );
            pacing_modes = {'realtime', 'accelerated', 'lockstep'};
            if flags == 34815   % every timing value and the pacing mode (the simulator's packets)
                v = timing_values;
                state.timing = struct('physics_timestep_seconds', v(1), 'control_period_seconds', v(2), ...
                    'publish_every_n_steps', v(3), 'pacing_mode', pacing_modes{double(datagram(14)) + 1}, ...
                    'session_wall_elapsed_seconds', v(4), 'session_sim_elapsed_seconds', v(5), ...
                    'sim_wall_skew_seconds', v(6), 'realtime_factor', v(7), 'step_wall_seconds', v(8), ...
                    'publish_wall_seconds', v(9), 'command_wait_wall_seconds', v(10));
            elseif bitand(flags, 32768)
                timing_fields = {'physics_timestep_seconds', 'control_period_seconds', 'publish_every_n_steps', ...
                    'session_wall_elapsed_seconds', 'session_sim_elapsed_seconds', 'sim_wall_skew_seconds', ...
                    'realtime_factor', 'step_wall_seconds', 'publish_wall_seconds', 'command_wait_wall_seconds'};
                timing = struct();
                for index = 1:numel(timing_fields)
                    if index == 4 && bitand(flags, 1024)   % pacing_mode precedes the session times, as in the JSON
                        timing.pacing_mode = pacing_modes{double(datagram(14)) + 1};
                    end
                    if bitand(flags, 2 ^ (index - 1))
                        timing.(timing_fields{index}) = timing_values(index);
                    end
                end
                state.timing = timing;
            end

            % The record's rows in the order of state_packets.py, built into the
            % structs jsondecode gives (a few whole-array calls: per-element
            % code costs more than jsondecode itself for small teams).
            cells = num2cell(values);
            contact_fields = {'count'; 'total_force_magnitude'; 'max_force_magnitude'; ...
                'total_normal_force'; 'max_normal_force'};
            groups = cell(4, num_uavs);   % left_wheel, right_wheel, surface, wall
            for group_index = 1:4
                groups(group_index, :) = num2cell(cell2struct(cells(21 + 5 * group_index + (1:5), :), contact_fields, 1));
            end
            contact_summaries = cell2struct([cells(22:26, :); groups], ...
                [contact_fields; {'left_wheel'; 'right_wheel'; 'surface'; 'wall'}], 1);
            columns = @(first, last) num2cell(values(first:last, :), 1);
            uavs = cell2struct([names; repmat({sim_time_and_rtf(1)}, 1, num_uavs); ...
                columns(1, 3); columns(4, 6); columns(7, 9); columns(10, 12); columns(13, 21); ...
                cells(3, :); cells(6, :); cells(9, :); ...
                repmat({sim_time_and_rtf(2)}, 1, num_uavs); repmat({instance_id}, 1, num_uavs); ...
                reshape(num2cell(contact_summaries), 1, []); repmat({[]}, 1, num_uavs)], ...
                {'name'; 'time'; 'position'; 'velocity'; 'angular_velocity_world'; 'angular_velocity_body'; ...
                'rotation_matrix'; 'z'; 'vz'; 'yaw_rate'; 'realtime_factor'; 'instance_id'; ...
                'contact_summary'; 'contacts'}, 1);

            if num_uavs == 1   % single-UAV packets carry the UAV's fields at the top level
                for field_name = fieldnames(uavs)'
                    state.(field_name{1}) = uavs.(field_name{1});
                end
                return;
            end
            state.time = sim_time_and_rtf(1);
            state.instance_id = instance_id;
            state.num_uavs = num_uavs;
            state.realtime_factor = sim_time_and_rtf(2);
            state.uavs = uavs;
        end

        function options = parse_command_options(default_fidelity_mode, args)
            options = struct( ...
                'sequence', [], ...
                'source_state_sequence', [], ...
                'wall_time_send_ns', [], ...
                'controller_compute_ms', [], ...
                'state_age_ms', [], ...
                'state_sequence_gap', [], ...
                'fidelity_mode', default_fidelity_mode, ...
                'body_wrench', [], ...
                'extensions', [] ...
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
                    case 'extensions'
                        is_valid = isempty(value) || (isstruct(value) && isscalar(value));
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
