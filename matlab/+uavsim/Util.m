classdef Util
    % UAVSIM.UTIL  Small struct/vector helpers shared across the MATLAB library.
    methods (Static)
        function value = get_struct_field(input_struct, field_name, default_value)
            value = default_value;
            if ~isstruct(input_struct) || ~isfield(input_struct, field_name)
                return;
            end
            value = input_struct.(field_name);
        end

        function value = get_optional_scalar(input_struct, field_name, default_value)
            value = default_value;
            if ~isstruct(input_struct) || ~isfield(input_struct, field_name)
                return;
            end
            value = double(input_struct.(field_name));
        end

        function value = get_optional_vector(input_struct, field_name, default_value, expected_length)
            value = default_value;
            if ~isstruct(input_struct) || ~isfield(input_struct, field_name)
                return;
            end
            candidate = reshape(double(input_struct.(field_name)), [], 1);
            if numel(candidate) ~= expected_length
                error('uavsim:invalidVector', '%s must have %d elements.', field_name, expected_length);
            end
            value = candidate;
        end

        function value = get_required_vector(input_struct, field_name, expected_length, context_label)
            if ~isfield(input_struct, field_name)
                error('uavsim:missingField', '%s.%s must be present.', context_label, field_name);
            end

            value = reshape(double(input_struct.(field_name)), [], 1);
            if numel(value) ~= expected_length
                error('uavsim:invalidVector', '%s.%s must have %d elements.', context_label, field_name, expected_length);
            end
        end

        function value = default_numeric(value)
            if isempty(value)
                value = NaN;
                return;
            end
            value = double(value);
        end

        function column_vector = state_vector(value)
            column_vector = reshape(double(value), [], 1);
        end

        function normalized = normalize_vector(vector, fallback)
            vector_norm = norm(vector);
            if vector_norm < 1e-6
                normalized = fallback;
                return;
            end

            normalized = vector / vector_norm;
        end

        function rotation_matrix = rotation_from_state(state)
            % state.rotation_matrix is row-major flattened; MATLAB reshape is
            % column-major, hence the transpose.
            rotation_matrix = reshape(double(state.rotation_matrix), [3, 3])';
        end

        function path_value = resolve_path_option(path_value, default_value)
            % Empty-tolerant path option: '' means "use the default".
            if isempty(path_value)
                path_value = default_value;
            end
            path_value = char(path_value);
        end

        function time_ns = wall_time_now_ns()
            time_ns = floor(posixtime(datetime('now', 'TimeZone', 'UTC')) * 1.0e9);
        end
    end
end
