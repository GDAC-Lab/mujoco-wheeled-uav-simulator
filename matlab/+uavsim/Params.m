classdef Params
    % UAVSIM.PARAMS  Loads vehicle_params.json into the MATLAB-side view.
    %
    % The returned struct mirrors what controllers need: total mass, gravity,
    % actuation limits, rotor geometry, controller/formation defaults, and
    % fidelity settings. The same JSON file drives the Python simulator, so
    % this parser is the MATLAB half of the shared-parameter contract.
    methods (Static)
        function vehicle_params = load(project_directory, varargin)
            parser = inputParser;
            addParameter(parser, 'params_path', fullfile(project_directory, 'vehicle_params.json'), @(value) ischar(value) || (isstring(value) && isscalar(value)));
            parse(parser, varargin{:});

            params_path = char(parser.Results.params_path);
            raw_params = jsondecode(fileread(params_path));
            % Overlay actuation.calibration_file before anything reads the
            % actuation block, as load_vehicle_params() does on the Python side.
            % In command_mode 'omega' the controller's thrust -> omega and the
            % simulator's omega -> thrust only cancel when both use the same kf.
            raw_params = uavsim.Params.apply_calibration_file(raw_params, params_path);

            vehicle_params = struct( ...
                'mass', uavsim.Params.parse_total_mass(raw_params), ...
                'gravity', abs(double(raw_params.simulation.gravity(3))), ...
                'arm_x', abs(double(raw_params.drone.arm.x)), ...
                'arm_y', abs(double(raw_params.drone.arm.y)), ...
                'yaw_moment_ratio', double(raw_params.actuation.yaw_moment_ratio), ...
                'max_rotor_thrust', double(raw_params.actuation.max_rotor_thrust), ...
                'thrust_coefficient', double(raw_params.actuation.thrust_coefficient), ...
                'wheel_friction', double(raw_params.drone.wheels.friction(1)), ...
                'command_mode', char(raw_params.actuation.command_mode), ...
                'controller', uavsim.Params.parse_controller_params(raw_params), ...
                'raw', raw_params, ...   % controller-specific blocks are parsed by the controllers themselves
                'rotors', uavsim.Params.parse_rotor_params(raw_params), ...
                'formation', uavsim.Params.parse_formation_params(raw_params), ...
                'fidelity', uavsim.Params.parse_fidelity_params(raw_params), ...
                'params_path', params_path ...
            );
        end

        function total_mass = parse_total_mass(raw_params)
            % drone.inertial_reference "total_vehicle": drone.mass already includes the wheels.
            % "body_only": total = body mass + 2 * wheel mass. A missing key keeps the
            % "body_only" default but warns, as the Python simulator does: reading a
            % whole-vehicle measurement as "body_only" counts the wheel mass twice.
            drone = raw_params.drone;
            if isfield(drone, 'mass')
                body_mass = double(drone.mass);
            else
                body_mass = double(drone.body_box.mass);
            end
            if isfield(drone, 'inertial_reference')
                reference = lower(strtrim(char(drone.inertial_reference)));
            else
                reference = 'body_only';
                wheel_mass = double(drone.wheels.mass);
                warning('uavsim:Params:inertialReferenceUnset', ...
                    ['drone.inertial_reference is not set; assuming "body_only": the model gets ' ...
                     'drone.mass %g kg plus two wheels of %g kg = %g kg in total. If drone.mass ' ...
                     'and drone.inertia describe the whole vehicle, set "total_vehicle"; ' ...
                     'otherwise set "body_only" explicitly to silence this warning.'], ...
                    body_mass, wheel_mass, body_mass + 2.0 * wheel_mass);
            end
            switch reference
                case 'total_vehicle'
                    total_mass = body_mass;
                case 'body_only'
                    total_mass = body_mass + 2.0 * double(drone.wheels.mass);
                otherwise
                    error('uavsim:Params:badInertialReference', ...
                        'drone.inertial_reference must be "body_only" or "total_vehicle" (got "%s").', reference);
            end
        end

        function raw_params = apply_calibration_file(raw_params, params_path)
            % MATLAB port of apply_calibration_file (wheeled_uav/calibration.py):
            % overlays the sim_params of the file named by
            % actuation.calibration_file (schema uav-propulsion-calibration/1)
            % and records raw_params.calibration_applied (file, source, applied).
            % A relative path resolves against the directory of the params file,
            % and null entries are skipped, so a calibration overrides only what
            % it measured. See docs/CALIBRATION.md.
            if ~isfield(raw_params, 'actuation') || ~isstruct(raw_params.actuation) ...
                    || ~isfield(raw_params.actuation, 'calibration_file')
                return;
            end
            raw_path = raw_params.actuation.calibration_file;
            if isempty(raw_path)
                return;   % null or "": no calibration referenced
            end
            calibration_path = char(raw_path);
            if ~uavsim.Util.is_absolute_path(calibration_path)
                calibration_path = fullfile(fileparts(char(params_path)), calibration_path);
            end
            if ~isfile(calibration_path)
                error('uavsim:Params:calibrationNotFound', ...
                    'actuation.calibration_file not found: %s', calibration_path);
            end

            document = jsondecode(fileread(calibration_path));
            schema = '';
            if isstruct(document) && isfield(document, 'schema') && ischar(document.schema)
                schema = document.schema;
            end
            if ~strcmp(schema, 'uav-propulsion-calibration/1')
                error('uavsim:Params:unsupportedCalibrationSchema', ...
                    'Unsupported calibration schema ''%s'' in %s; expected ''uav-propulsion-calibration/1''', ...
                    schema, calibration_path);
            end
            if ~isfield(document, 'sim_params') || ~isstruct(document.sim_params)
                error('uavsim:Params:invalidCalibration', ...
                    'Calibration file %s has no ''sim_params'' object', calibration_path);
            end
            sim_params = document.sim_params;
            applied = struct();

            value = uavsim.Params.get_calibration_value(sim_params, 'thrust_coefficient');
            if ~isempty(value)
                if value <= 0.0
                    error('uavsim:Params:invalidCalibration', 'calibration sim_params.thrust_coefficient must be positive');
                end
                raw_params.actuation.thrust_coefficient = value;
                applied.thrust_coefficient = value;
            end

            value = uavsim.Params.get_calibration_value(sim_params, 'yaw_moment_ratio');
            if ~isempty(value)
                if value <= 0.0
                    error('uavsim:Params:invalidCalibration', 'calibration sim_params.yaw_moment_ratio must be positive');
                end
                raw_params.actuation.yaw_moment_ratio = value;
                % The bench test measures one km/kf for the whole propulsion set,
                % and a rotor's own yaw_moment_ratio wins over the global one, so
                % align those too: a stale per-rotor value would silently win.
                if isfield(raw_params.actuation, 'rotors')
                    raw_params.actuation.rotors = uavsim.Params.set_rotor_yaw_moment_ratio(raw_params.actuation.rotors, value);
                end
                applied.yaw_moment_ratio = value;
            end

            value = uavsim.Params.get_calibration_value(sim_params, 'motor_tau_ms');
            if ~isempty(value)
                if value < 0.0
                    error('uavsim:Params:invalidCalibration', 'calibration sim_params.motor_tau_ms must be >= 0');
                end
                if ~isfield(raw_params, 'actuator_dynamics')
                    raw_params.actuator_dynamics = struct();
                end
                if isstruct(raw_params.actuator_dynamics)
                    raw_params.actuator_dynamics.motor_tau_ms = value;
                    applied.motor_tau_ms = value;
                end
            end

            source = struct();
            if isfield(document, 'source') && isstruct(document.source) && isscalar(document.source)
                source = document.source;
            end
            raw_params.calibration_applied = struct('file', calibration_path, 'source', source, 'applied', applied);

            [~, file_name, file_extension] = fileparts(calibration_path);
            applied_names = fieldnames(applied);
            if isempty(applied_names)
                fprintf('calibration: %s%s has no applicable sim_params; nothing overridden\n', file_name, file_extension);
            else
                applied_texts = cell(1, numel(applied_names));
                for name_index = 1:numel(applied_names)
                    applied_texts{name_index} = sprintf('%s=%g', applied_names{name_index}, applied.(applied_names{name_index}));
                end
                fprintf('calibration: %s%s -> %s\n', file_name, file_extension, strjoin(applied_texts, ', '));
            end
        end

        function value = get_calibration_value(sim_params, field_name)
            % [] when the entry is absent or null (jsondecode gives []).
            value = [];
            if ~isfield(sim_params, field_name) || isempty(sim_params.(field_name))
                return;
            end
            candidate = sim_params.(field_name);
            if ~isnumeric(candidate) || ~isscalar(candidate)
                error('uavsim:Params:invalidCalibration', ...
                    'calibration sim_params.%s must be a number or null', field_name);
            end
            value = double(candidate);
        end

        function rotors = set_rotor_yaw_moment_ratio(rotors, value)
            % Only rotors that carry their own yaw_moment_ratio change; the others
            % already inherit actuation.yaw_moment_ratio. jsondecode gives a struct
            % array when every rotor has the same fields and a cell array otherwise.
            if isstruct(rotors)
                if isfield(rotors, 'yaw_moment_ratio')
                    [rotors.yaw_moment_ratio] = deal(value);
                end
            elseif iscell(rotors)
                for rotor_index = 1:numel(rotors)
                    if isstruct(rotors{rotor_index}) && isfield(rotors{rotor_index}, 'yaw_moment_ratio')
                        rotors{rotor_index}.yaw_moment_ratio = value;
                    end
                end
            end
        end

        function [allocation_matrix, mixer] = build_allocation_and_mixer(vehicle_params)
            % Allocation columns derived from actual rotor geometry:
            %   column_i = [thrust_axis_z; p_i x axis_i + spin_i * kappa_i * axis_i]
            % so the wrench-to-rotor map always matches the simulator actuator
            % ordering, including tilted rotors and arbitrary JSON rotor order.
            rotors = vehicle_params.rotors;
            allocation_matrix = zeros(4, numel(rotors));
            for rotor_index = 1:numel(rotors)
                rotor = rotors(rotor_index);
                axis_body = reshape(double(rotor.thrust_axis_body), 3, 1);
                position_body = reshape(double(rotor.position_body), 3, 1);
                moment_column = cross(position_body, axis_body) ...
                    + double(rotor.spin_sign) * double(rotor.yaw_moment_ratio) * axis_body;
                allocation_matrix(:, rotor_index) = [axis_body(3); moment_column];
            end
            mixer = pinv(allocation_matrix);
        end

        function controller_params = parse_controller_params(raw_params)
            controller_params = struct( ...
                'desired_heading', [1.0; 0.0; 0.0], ...
                'position_gain', [3.0; 3.0; 6.0], ...
                'velocity_gain', [2.2; 2.2; 4.0], ...
                'attitude_gain', [0.8; 0.8; 0.25], ...
                'angular_velocity_gain', [0.12; 0.12; 0.08], ...
                'position_error_limit_m', uavsim.Control.DEFAULT_POSITION_ERROR_LIMIT_M, ...
                'max_tilt_deg', uavsim.Control.DEFAULT_MAX_TILT_DEG ...
            );

            if ~isfield(raw_params, 'controller')
                return;
            end

            raw_controller = raw_params.controller;
            controller_params.desired_heading = uavsim.Util.get_optional_vector(raw_controller, 'desired_heading', controller_params.desired_heading, 3);
            controller_params.position_gain = uavsim.Util.get_optional_vector(raw_controller, 'position_gain', controller_params.position_gain, 3);
            controller_params.velocity_gain = uavsim.Util.get_optional_vector(raw_controller, 'velocity_gain', controller_params.velocity_gain, 3);
            controller_params.attitude_gain = uavsim.Util.get_optional_vector(raw_controller, 'attitude_gain', controller_params.attitude_gain, 3);
            controller_params.angular_velocity_gain = uavsim.Util.get_optional_vector(raw_controller, 'angular_velocity_gain', controller_params.angular_velocity_gain, 3);
            controller_params.position_error_limit_m = uavsim.Util.get_optional_scalar(raw_controller, 'position_error_limit_m', controller_params.position_error_limit_m);
            controller_params.max_tilt_deg = uavsim.Util.get_optional_scalar(raw_controller, 'max_tilt_deg', controller_params.max_tilt_deg);
        end

        function rotor_params = parse_rotor_params(raw_params)
            % No abs(): a negative actuation.yaw_moment_ratio is a legitimate
            % global spin-convention flip and must build the SAME allocation
            % matrix as the Python side (model/builder.py uses the raw value).
            default_yaw_moment_ratio = double(raw_params.actuation.yaw_moment_ratio);
            if isfield(raw_params.actuation, 'rotors') && ~isempty(raw_params.actuation.rotors)
                raw_rotors = raw_params.actuation.rotors;
                if iscell(raw_rotors)
                    % jsondecode yields a cell array when the rotor entries have
                    % different field sets, which is legal here (spin_sign and
                    % yaw_moment_ratio are optional per rotor).
                    raw_rotor_list = raw_rotors(:);
                elseif isstruct(raw_rotors)
                    raw_rotor_list = num2cell(raw_rotors(:));
                else
                    error('uavsim:invalidRotors', 'actuation.rotors must be an array of rotor objects.');
                end
                if numel(raw_rotor_list) ~= 4
                    % Mirrors the Python builder; the logger schema also
                    % hard-codes 4 rotor channels.
                    error('uavsim:invalidRotors', 'actuation.rotors must define exactly 4 rotors (got %d).', numel(raw_rotor_list));
                end

                rotor_params = repmat(struct( ...
                    'name', '', ...
                    'position_body', zeros(3, 1), ...
                    'thrust_axis_body', zeros(3, 1), ...
                    'yaw_moment_ratio', 0.0, ...
                    'spin_sign', 1.0 ...
                ), numel(raw_rotor_list), 1);

                for rotor_index = 1:numel(raw_rotor_list)
                    raw_rotor = raw_rotor_list{rotor_index};
                    if ~isfield(raw_rotor, 'name')
                        error('uavsim:invalidRotors', 'actuation.rotors(%d) must have a name.', rotor_index);
                    end
                    position_body = uavsim.Util.get_required_vector(raw_rotor, 'position_body', 3, sprintf('actuation.rotors(%d)', rotor_index));
                    thrust_axis_body = uavsim.Util.normalize_vector( ...
                        uavsim.Util.get_required_vector(raw_rotor, 'thrust_axis_body', 3, sprintf('actuation.rotors(%d)', rotor_index)), ...
                        [0.0; 0.0; 1.0] ...
                    );
                    spin_sign = uavsim.Util.get_optional_scalar(raw_rotor, 'spin_sign', 1.0);
                    if abs(spin_sign) < 1.0e-9
                        error('uavsim:invalidRotors', 'actuation.rotors(%d).spin_sign must be non-zero.', rotor_index);
                    end

                    rotor_params(rotor_index) = struct( ...
                        'name', char(raw_rotor.name), ...
                        'position_body', position_body, ...
                        'thrust_axis_body', thrust_axis_body, ...
                        'yaw_moment_ratio', uavsim.Util.get_optional_scalar(raw_rotor, 'yaw_moment_ratio', default_yaw_moment_ratio), ...
                        'spin_sign', sign(spin_sign) ...
                    );
                end
                return;
            end

            arm_x = abs(double(raw_params.drone.arm.x));
            arm_y = abs(double(raw_params.drone.arm.y));
            propeller_z = double(raw_params.drone.propeller.z);
            rotor_params = [ ...
                struct('name', 'fr', 'position_body', [arm_x; -arm_y; propeller_z], 'thrust_axis_body', [0.0; 0.0; 1.0], 'yaw_moment_ratio', default_yaw_moment_ratio, 'spin_sign', 1.0); ...
                struct('name', 'fl', 'position_body', [arm_x; arm_y; propeller_z], 'thrust_axis_body', [0.0; 0.0; 1.0], 'yaw_moment_ratio', default_yaw_moment_ratio, 'spin_sign', -1.0); ...
                struct('name', 'br', 'position_body', [-arm_x; -arm_y; propeller_z], 'thrust_axis_body', [0.0; 0.0; 1.0], 'yaw_moment_ratio', default_yaw_moment_ratio, 'spin_sign', -1.0); ...
                struct('name', 'bl', 'position_body', [-arm_x; arm_y; propeller_z], 'thrust_axis_body', [0.0; 0.0; 1.0], 'yaw_moment_ratio', default_yaw_moment_ratio, 'spin_sign', 1.0) ...
            ];
        end

        function formation_params = parse_formation_params(raw_params)
            controller_defaults = uavsim.Params.parse_controller_params(raw_params);
            formation_params = struct( ...
                'num_uavs', 3, ...
                'spawn_radius', 1.5, ...
                'base_height', 1.5, ...
                'centroid_target_xy', [0.0; 0.0], ...
                'formation_radius', 1.5, ...
                'centroid_gain', 0.8, ...
                'formation_gain', 1.2, ...
                'duration_seconds', 20.0, ...
                'idle_sleep_seconds', 0.001, ...
                'status_display_interval', 2.0, ...
                'desired_heading', controller_defaults.desired_heading, ...
                'position_gain', controller_defaults.position_gain, ...
                'velocity_gain', controller_defaults.velocity_gain, ...
                'attitude_gain', controller_defaults.attitude_gain, ...
                'angular_velocity_gain', controller_defaults.angular_velocity_gain, ...
                'position_error_limit_m', controller_defaults.position_error_limit_m, ...
                'max_tilt_deg', controller_defaults.max_tilt_deg ...
            );

            if ~isfield(raw_params, 'formation')
                return;
            end

            raw_formation = raw_params.formation;
            formation_params.num_uavs = uavsim.Util.get_optional_scalar(raw_formation, 'num_uavs', formation_params.num_uavs);
            formation_params.spawn_radius = uavsim.Util.get_optional_scalar(raw_formation, 'spawn_radius', formation_params.spawn_radius);
            formation_params.base_height = uavsim.Util.get_optional_scalar(raw_formation, 'base_height', formation_params.base_height);
            formation_params.formation_radius = uavsim.Util.get_optional_scalar(raw_formation, 'formation_radius', formation_params.formation_radius);
            formation_params.centroid_gain = uavsim.Util.get_optional_scalar(raw_formation, 'centroid_gain', formation_params.centroid_gain);
            formation_params.formation_gain = uavsim.Util.get_optional_scalar(raw_formation, 'formation_gain', formation_params.formation_gain);
            formation_params.duration_seconds = uavsim.Util.get_optional_scalar(raw_formation, 'duration_seconds', formation_params.duration_seconds);
            formation_params.idle_sleep_seconds = uavsim.Util.get_optional_scalar(raw_formation, 'idle_sleep_seconds', formation_params.idle_sleep_seconds);
            formation_params.status_display_interval = uavsim.Util.get_optional_scalar(raw_formation, 'status_display_interval', formation_params.status_display_interval);
            formation_params.centroid_target_xy = uavsim.Util.get_optional_vector(raw_formation, 'centroid_target_xy', formation_params.centroid_target_xy, 2);
            formation_params.desired_heading = uavsim.Util.get_optional_vector(raw_formation, 'desired_heading', formation_params.desired_heading, 3);
            formation_params.position_gain = uavsim.Util.get_optional_vector(raw_formation, 'position_gain', formation_params.position_gain, 3);
            formation_params.velocity_gain = uavsim.Util.get_optional_vector(raw_formation, 'velocity_gain', formation_params.velocity_gain, 3);
            formation_params.attitude_gain = uavsim.Util.get_optional_vector(raw_formation, 'attitude_gain', formation_params.attitude_gain, 3);
            formation_params.angular_velocity_gain = uavsim.Util.get_optional_vector(raw_formation, 'angular_velocity_gain', formation_params.angular_velocity_gain, 3);
            formation_params.position_error_limit_m = uavsim.Util.get_optional_scalar(raw_formation, 'position_error_limit_m', formation_params.position_error_limit_m);
            formation_params.max_tilt_deg = uavsim.Util.get_optional_scalar(raw_formation, 'max_tilt_deg', formation_params.max_tilt_deg);
        end

        function fidelity_params = parse_fidelity_params(raw_params)
            fidelity_mode = 'baseline';
            if isfield(raw_params, 'fidelity_mode')
                if ischar(raw_params.fidelity_mode) || (isstring(raw_params.fidelity_mode) && isscalar(raw_params.fidelity_mode))
                    fidelity_mode = char(raw_params.fidelity_mode);
                elseif isstruct(raw_params.fidelity_mode) && isfield(raw_params.fidelity_mode, 'mode')
                    fidelity_mode = char(raw_params.fidelity_mode.mode);
                end
            end

            network_section = struct();
            if isfield(raw_params, 'network_fidelity') && isstruct(raw_params.network_fidelity)
                network_section = raw_params.network_fidelity;
            end

            fidelity_params = struct( ...
                'mode', fidelity_mode, ...
                'network', struct( ...
                    'enabled', logical(uavsim.Util.get_struct_field(network_section, 'enabled', false)), ...
                    'state_tx_latency_ms', uavsim.Util.get_optional_scalar(network_section, 'state_tx_latency_ms', 0.0), ...
                    'command_rx_latency_ms', uavsim.Util.get_optional_scalar(network_section, 'command_rx_latency_ms', 0.0), ...
                    'packet_loss_percent', uavsim.Util.get_optional_scalar(network_section, 'packet_loss_percent', 0.0), ...
                    'jitter_std_dev_ms', uavsim.Util.get_optional_scalar(network_section, 'jitter_std_dev_ms', 0.0), ...
                    'stale_command_threshold_ms', uavsim.Util.default_numeric(uavsim.Util.get_struct_field(network_section, 'stale_command_threshold_ms', NaN)), ...
                    'stale_command_policy', char(uavsim.Util.get_struct_field(network_section, 'stale_command_policy', 'hold-last-command')) ...
                ) ...
            );
        end
    end
end
