classdef Session
    % UAVSIM.SESSION  Controller session bootstrap: params, socket, ports, launch.
    %
    % Typical use in a controller entry point:
    %   session = uavsim.Session.start(project_dir, runtime_options, ...
    %       'params_path', params_path);
    %   cleanup = onCleanup(@() uavsim.Session.cleanup_socket(session.controller_socket));
    %   ... loop using session.vehicle_params / session.controller_socket ...
    methods (Static)
        function controller_session = start(project_directory, runtime_options, varargin)
            parser = inputParser;
            addParameter(parser, 'target_ip', '127.0.0.1', @(value) ischar(value) || (isstring(value) && isscalar(value)));
            addParameter(parser, 'simulator_root', project_directory, @(value) ischar(value) || (isstring(value) && isscalar(value)));
            addParameter(parser, 'params_path', fullfile(project_directory, 'vehicle_params.json'), @(value) ischar(value) || (isstring(value) && isscalar(value)));
            addParameter(parser, 'generated_xml_directory', fullfile(project_directory, 'build', 'generated_xml'), @(value) ischar(value) || (isstring(value) && isscalar(value)));
            parse(parser, varargin{:});

            vehicle_params = uavsim.Params.load(project_directory, 'params_path', char(parser.Results.params_path));
            instance_options = uavsim.Session.build_instance_options(runtime_options.instance_id);
            uavsim.Session.release_stale_controller_socket(instance_options.controller_local_port);
            uavsim.Launch.assert_udp_port_available(instance_options.controller_local_port, instance_options.instance_id);
            controller_socket = udpport("datagram", "IPv4", "LocalPort", instance_options.controller_local_port, ...
                "OutputDatagramSize", 65507);
            % OutputDatagramSize default is 512 bytes: larger command payloads
            % (e.g. 10-UAV rotor commands) would be silently fragmented into
            % multiple datagrams, which the simulator cannot reassemble
            % (json "Extra data" crash).

            try
                simulator_options = uavsim.Session.build_simulator_options( ...
                    project_directory, ...
                    instance_options, ...
                    'simulator_root', char(parser.Results.simulator_root), ...
                    'params_path', char(parser.Results.params_path), ...
                    'generated_xml_directory', char(parser.Results.generated_xml_directory) ...
                );
                simulator_options.auto_launch = uavsim.Util.get_struct_field(runtime_options, 'auto_launch', false);
                simulator_options.shutdown_on_exit = uavsim.Util.get_struct_field(runtime_options, 'shutdown_on_exit', false);
                simulator_options.num_uavs = uavsim.Util.get_struct_field(runtime_options, 'num_uavs', simulator_options.num_uavs);
                simulator_options.spawn_radius = uavsim.Util.get_struct_field(runtime_options, 'spawn_radius', simulator_options.spawn_radius);
                simulator_options.wait_for_startup_seconds = uavsim.Util.get_struct_field(runtime_options, 'wait_for_startup_seconds', simulator_options.wait_for_startup_seconds);
                simulator_options.headless = uavsim.Util.get_struct_field(runtime_options, 'headless', false);
                simulator_options.simulation_duration_seconds = uavsim.Util.get_struct_field(runtime_options, 'simulation_duration_seconds', simulator_options.simulation_duration_seconds);
                simulator_options.record_path = uavsim.Util.get_struct_field(runtime_options, 'record_path', simulator_options.record_path);
                simulator_process_id = uavsim.Launch.launch_simulator_if_requested(simulator_options);
            catch bootstrap_error
                % No onCleanup handler owns the socket yet; without this a
                % failed launch leaks the bound port until MATLAB exits.
                delete(controller_socket);
                rethrow(bootstrap_error);
            end

            controller_session = struct( ...
                'vehicle_params', vehicle_params, ...
                'instance_options', instance_options, ...
                'controller_socket', controller_socket, ...
                'target_ip', char(parser.Results.target_ip), ...
                'target_port', instance_options.simulator_receive_port, ...
                'simulator_options', simulator_options, ...
                'simulator_process_id', simulator_process_id ...
            );
        end

        function instance_options = build_instance_options(instance_id)
            arguments
                instance_id (1, 1) double {mustBeInteger, mustBeNonnegative} = 0
            end

            simulator_receive_port = 5000 + 2 * instance_id;
            controller_local_port = simulator_receive_port + 1;
            file_suffix = '';
            if instance_id ~= 0
                file_suffix = sprintf('_i%d', instance_id);
            end

            instance_options = struct( ...
                'instance_id', double(instance_id), ...
                'label', sprintf('instance=%d', instance_id), ...
                'simulator_receive_port', simulator_receive_port, ...
                'controller_local_port', controller_local_port, ...
                'file_suffix', file_suffix ...
            );
        end

        function simulator_options = build_simulator_options(project_directory, instance_options, varargin)
            parser = inputParser;
            addParameter(parser, 'simulator_root', project_directory, @(value) ischar(value) || (isstring(value) && isscalar(value)));
            addParameter(parser, 'params_path', fullfile(project_directory, 'vehicle_params.json'), @(value) ischar(value) || (isstring(value) && isscalar(value)));
            addParameter(parser, 'generated_xml_directory', fullfile(project_directory, 'build', 'generated_xml'), @(value) ischar(value) || (isstring(value) && isscalar(value)));
            parse(parser, varargin{:});

            simulator_root = char(parser.Results.simulator_root);
            simulator_options = struct( ...
                'auto_launch', false, ...
                'wait_for_startup_seconds', 3.0, ...
                'simulator_receive_port', instance_options.simulator_receive_port, ...
                'shutdown_on_exit', false, ...
                'headless', false, ...
                'simulation_duration_seconds', NaN, ...
                'record_path', '', ...
                'instance_id', instance_options.instance_id, ...
                'num_uavs', 1, ...
                'spawn_radius', 1.5, ...
                'working_directory', project_directory, ...
                'simulator_root', simulator_root, ...
                'params_path', char(parser.Results.params_path), ...
                'generated_xml_directory', char(parser.Results.generated_xml_directory) ...
            );
        end

        function release_stale_controller_socket(local_port)
            if exist('udpportfind', 'file') ~= 2
                return;
            end

            try
                stale_sockets = udpportfind("LocalPort", local_port);
            catch
                stale_sockets = [];
            end

            if isempty(stale_sockets)
                return;
            end

            fprintf('Releasing stale UDP socket on port %d before controller startup.\n', local_port);
            try
                delete(stale_sockets);
            catch
            end
        end

        function cleanup_socket(controller_socket)
            try
                delete(controller_socket);
            catch
            end
        end

        function display_logging_behavior(logger)
            logging_options = logger.get_options();
            is_periodic = strcmp(logging_options.save_mode, 'periodic') || strcmp(logging_options.save_mode, 'periodic_and_finalize');
            if is_periodic
                fprintf('Logging policy: mode=%s, interval=%.2f s, path=%s\n', ...
                    logging_options.save_mode, ...
                    logging_options.periodic_interval_seconds, ...
                    logger.get_file_path() ...
                );
            else
                fprintf('Logging policy: mode=%s, path=%s\n', ...
                    logging_options.save_mode, ...
                    logger.get_file_path() ...
                );
            end
        end

        function finalize_controller_run(logger)
            logging_options = logger.get_options();
            supports_finalize = strcmp(logging_options.save_mode, 'finalize') || strcmp(logging_options.save_mode, 'periodic_and_finalize');
            if ~supports_finalize
                return;
            end

            logger.finalize();
            if logging_options.print_save_events
                fprintf('Simulation log saved at shutdown -> %s\n', logger.get_file_path());
            end
        end
    end
end
