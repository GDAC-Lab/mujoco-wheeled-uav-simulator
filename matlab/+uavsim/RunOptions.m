classdef RunOptions
    % UAVSIM.RUNOPTIONS  Shared name/value option parsing for controller entries.
    %
    % Every sample controller accepts the same base options (instance_id,
    % auto_launch, duration_seconds, paths, ...). parse() handles those in one
    % place; controllers add their own options via 'extra_parameters', a cell
    % array of {name, default, validator} rows.
    methods (Static)
        function runtime_options = parse(varargin_cell, extra_parameters)
            if nargin < 2
                extra_parameters = {};
            end

            parser = inputParser;
            addParameter(parser, 'instance_id', 0, @(value) validateattributes(value, {'numeric'}, {'scalar', 'integer', 'nonnegative'}));
            addParameter(parser, 'duration_seconds', inf, @(value) (isnumeric(value) && isscalar(value) && value > 0) || isinf(value));
            addParameter(parser, 'wait_for_startup_seconds', 3.0, @(value) validateattributes(value, {'numeric'}, {'scalar', 'positive'}));
            % 30 s (not inf): a missing simulator must produce a diagnostic,
            % not a silent infinite spin. Generous enough to start the
            % controller first and then launch the simulator by hand; pass
            % 'state_timeout_seconds', inf to opt out.
            addParameter(parser, 'state_timeout_seconds', 30.0, @(value) (isnumeric(value) && isscalar(value) && value > 0) || isinf(value));
            addParameter(parser, 'headless', false, @(value) islogical(value) || (isnumeric(value) && isscalar(value)));
            addParameter(parser, 'simulation_duration_seconds', NaN, @(value) (isnumeric(value) && isscalar(value)) || isempty(value));
            addParameter(parser, 'auto_launch', false, @(value) islogical(value) || (isnumeric(value) && isscalar(value)));
            addParameter(parser, 'shutdown_on_exit', false, @(value) islogical(value) || (isnumeric(value) && isscalar(value)));
            addParameter(parser, 'record_path', '', @(value) ischar(value) || (isstring(value) && isscalar(value)));
            addParameter(parser, 'simulator_root', '', @(value) ischar(value) || (isstring(value) && isscalar(value)));
            addParameter(parser, 'params_path', '', @(value) ischar(value) || (isstring(value) && isscalar(value)));
            addParameter(parser, 'generated_xml_directory', '', @(value) ischar(value) || (isstring(value) && isscalar(value)));
            for extra_index = 1:size(extra_parameters, 1)
                addParameter(parser, extra_parameters{extra_index, 1}, extra_parameters{extra_index, 2}, extra_parameters{extra_index, 3});
            end
            parse(parser, varargin_cell{:});

            runtime_options = parser.Results;
            runtime_options.instance_id = double(runtime_options.instance_id);
            runtime_options.duration_seconds = double(runtime_options.duration_seconds);
            runtime_options.wait_for_startup_seconds = double(runtime_options.wait_for_startup_seconds);
            runtime_options.state_timeout_seconds = double(runtime_options.state_timeout_seconds);
            runtime_options.headless = logical(runtime_options.headless);
            runtime_options.simulation_duration_seconds = double(runtime_options.simulation_duration_seconds);
            runtime_options.auto_launch = logical(runtime_options.auto_launch);
            runtime_options.shutdown_on_exit = logical(runtime_options.shutdown_on_exit);
            runtime_options.record_path = char(runtime_options.record_path);
            runtime_options.simulator_root = char(runtime_options.simulator_root);
            runtime_options.params_path = char(runtime_options.params_path);
            runtime_options.generated_xml_directory = char(runtime_options.generated_xml_directory);
        end

        function logging_options = build_logging_options(file_prefix, instance_options, varargin)
            parser = inputParser;
            % Strict membership: a typo'd save_mode would otherwise disable
            % BOTH the periodic and the finalize save paths and the run would
            % silently produce no log file at all.
            addParameter(parser, 'save_mode', 'finalize', @(value) any(strcmp(char(value), {'finalize', 'periodic', 'periodic_and_finalize', 'none'})));
            addParameter(parser, 'periodic_interval_seconds', 30.0, @(value) isnumeric(value) && isscalar(value));
            addParameter(parser, 'print_save_events', true, @(value) islogical(value) || (isnumeric(value) && isscalar(value)));
            addParameter(parser, 'directory_name', 'logs', @(value) ischar(value) || (isstring(value) && isscalar(value)));
            parse(parser, varargin{:});

            logging_options = struct( ...
                'save_mode', char(parser.Results.save_mode), ...
                'periodic_interval_seconds', double(parser.Results.periodic_interval_seconds), ...
                'print_save_events', logical(parser.Results.print_save_events), ...
                'directory_name', char(parser.Results.directory_name), ...
                'file_prefix', [char(file_prefix) instance_options.file_suffix] ...
            );
        end

        function should_save = should_save_log_periodically(logging_options, simulation_time, next_log_save_time)
            supports_periodic = strcmp(logging_options.save_mode, 'periodic') || strcmp(logging_options.save_mode, 'periodic_and_finalize');
            has_valid_interval = logging_options.periodic_interval_seconds > 0.0;
            should_save = supports_periodic && has_valid_interval && simulation_time >= next_log_save_time;
        end
    end
end
