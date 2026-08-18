classdef Control
    % UAVSIM.CONTROL  Geometric control building blocks shared by the samples.
    %
    % The hover controller is decomposed so the pieces can be reused:
    %   hover_desired_force    world PD + gravity feedforward (+ error saturation)
    %   saturate_desired_force vertical-force floor + tilt cap for large errors
    %   attitude_from_force    desired rotation from a force direction + heading
    %   attitude_moments       SO(3) PD attitude tracking moments
    %   wrench_to_rotor_thrusts  [f; M] through the mixer with saturation
    % compute_hover_control composes them and matches the numerical behavior of
    % the Python reference controller (wheeled_uav/controllers/hover.py).
    properties (Constant)
        % Fraction of hover force kept as the minimum vertical component while
        % the tilt clamp is active; bounds commanded descent to (1 - 0.25) g
        % and keeps the desired body z-axis pointing up. Mirrors
        % _MIN_VERTICAL_FORCE_FACTOR in wheeled_uav/controllers/hover.py.
        MIN_VERTICAL_FORCE_FACTOR = 0.25;
        % Defaults for the large-displacement safety clamps (<= 0 disables).
        DEFAULT_POSITION_ERROR_LIMIT_M = 1.5;
        DEFAULT_MAX_TILT_DEG = 35.0;
    end
    methods (Static)
        function desired_force = hover_desired_force(state, target_position, mass, gravity, position_gain, velocity_gain, position_error_limit)
            if nargin < 7
                % Standalone building-block use keeps the raw (unsaturated) PD
                % by design; the composed compute_hover_control always passes
                % an explicit limit and defaults it to
                % DEFAULT_POSITION_ERROR_LIMIT_M instead.
                position_error_limit = 0.0;
            end
            position = uavsim.Util.state_vector(state.position);
            velocity = uavsim.Util.state_vector(state.velocity);
            position_error = target_position - position;
            error_norm = norm(position_error);
            if position_error_limit > 0.0 && error_norm > position_error_limit
                % Saturate the error norm so far-away targets (e.g. a vehicle
                % dragged in the viewer) command a bounded pull instead of an
                % arbitrarily aggressive maneuver.
                position_error = position_error * (position_error_limit / error_norm);
            end
            velocity_error = -velocity;
            desired_force = position_gain .* position_error ...
                + velocity_gain .* velocity_error ...
                + [0.0; 0.0; mass * gravity];
        end

        function clamped_force = saturate_desired_force(desired_force, mass, gravity, max_tilt_deg)
            % Floor the vertical force component and cap the tilt of the
            % desired thrust vector. Guarantees the desired body z-axis stays
            % within max_tilt_deg of vertical and never points down, so
            % recovery from a large displacement is a bounded, upright
            % maneuver instead of a flip. Disabled when max_tilt_deg <= 0;
            % values >= 90 keep only the vertical floor (tan would go negative
            % and flip the lateral force instead of capping it).
            clamped_force = desired_force;
            if max_tilt_deg <= 0.0
                return;
            end
            min_vertical_force = uavsim.Control.MIN_VERTICAL_FORCE_FACTOR * mass * gravity;
            clamped_force(3) = max(clamped_force(3), min_vertical_force);
            if max_tilt_deg >= 90.0
                return;
            end
            max_horizontal_force = clamped_force(3) * tan(deg2rad(max_tilt_deg));
            horizontal_norm = norm(clamped_force(1:2));
            if horizontal_norm > max_horizontal_force
                clamped_force(1:2) = clamped_force(1:2) * (max_horizontal_force / horizontal_norm);
            end
        end

        function desired_rotation = attitude_from_force(desired_force, desired_heading)
            % Desired body axes: z along the force, x as close as possible to
            % the heading, y completing the right-handed frame.
            desired_heading = uavsim.Util.normalize_vector(desired_heading, [1.0; 0.0; 0.0]);
            desired_body_z = uavsim.Util.normalize_vector(desired_force, [0.0; 0.0; 1.0]);
            desired_body_y = cross(desired_body_z, desired_heading);
            if norm(desired_body_y) < 1e-6
                desired_body_y = cross(desired_body_z, [0.0; 1.0; 0.0]);
            end
            desired_body_y = desired_body_y / norm(desired_body_y);
            desired_body_x = cross(desired_body_y, desired_body_z);
            desired_body_x = desired_body_x / norm(desired_body_x);
            desired_rotation = [desired_body_x, desired_body_y, desired_body_z];
        end

        function moment_command = attitude_moments(rotation_matrix, angular_velocity, desired_rotation, attitude_gain, angular_velocity_gain)
            % SO(3) PD: e_R = 0.5 * (R_d' R - R' R_d)^vee.
            attitude_error_matrix = 0.5 * (desired_rotation' * rotation_matrix - rotation_matrix' * desired_rotation);
            attitude_error = [attitude_error_matrix(3, 2); attitude_error_matrix(1, 3); attitude_error_matrix(2, 1)];
            moment_command = -attitude_gain .* attitude_error - angular_velocity_gain .* angular_velocity;
        end

        function rotor_thrusts = wrench_to_rotor_thrusts(collective_thrust, moment_command, mixer, max_rotor_thrust)
            wrench = [collective_thrust; moment_command];
            rotor_thrusts = min(max_rotor_thrust, max(0.0, mixer * wrench));
        end

        function rotor_thrusts = compute_hover_control(state, target_position, desired_heading, mass, gravity, position_gain, velocity_gain, attitude_gain, angular_velocity_gain, mixer, max_rotor_thrust, position_error_limit_m, max_tilt_deg)
            % Trailing safety-clamp args are optional; omitting them applies
            % the recommended defaults (see the Constant block). Pass <= 0 to
            % disable a clamp explicitly.
            if nargin < 12
                position_error_limit_m = uavsim.Control.DEFAULT_POSITION_ERROR_LIMIT_M;
            end
            if nargin < 13
                max_tilt_deg = uavsim.Control.DEFAULT_MAX_TILT_DEG;
            end
            rotation_matrix = uavsim.Util.rotation_from_state(state);
            angular_velocity = uavsim.Util.state_vector(state.angular_velocity_body);

            desired_force = uavsim.Control.hover_desired_force(state, target_position, mass, gravity, position_gain, velocity_gain, position_error_limit_m);
            desired_force = uavsim.Control.saturate_desired_force(desired_force, mass, gravity, max_tilt_deg);
            collective_thrust = max(0.0, dot(desired_force, rotation_matrix(:, 3)));
            desired_rotation = uavsim.Control.attitude_from_force(desired_force, desired_heading);
            moment_command = uavsim.Control.attitude_moments(rotation_matrix, angular_velocity, desired_rotation, attitude_gain, angular_velocity_gain);
            rotor_thrusts = uavsim.Control.wrench_to_rotor_thrusts(collective_thrust, moment_command, mixer, max_rotor_thrust);
        end
    end
end
