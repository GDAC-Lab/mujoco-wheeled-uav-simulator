classdef Launch
    % UAVSIM.LAUNCH  Optional MuJoCo simulator launch/shutdown from MATLAB.
    %
    % Prefers the .venv interpreter of the simulator checkout when present
    % (python -m wheeled_uav.cli), otherwise falls back to `uv run --project`.
    % Auto-launch stays opt-in; running the simulator in its own terminal is
    % the recommended default for clearer separation and debugging.
    methods (Static)
        function simulator_process_id = launch_simulator_if_requested(simulator_options)
            simulator_process_id = [];
            if ~simulator_options.auto_launch
                return;
            end

            if uavsim.Launch.does_udp_port_exist(simulator_options.simulator_receive_port)
                fprintf('MuJoCo simulator appears to be already running on UDP port %d.\n', simulator_options.simulator_receive_port);
                return;
            end

            [command_text, launch_mode] = uavsim.Launch.build_simulator_launch_command(simulator_options);
            fprintf('Launching MuJoCo simulator from MATLAB using %s.\n', launch_mode);

            [status, command_output] = system(command_text);
            if status ~= 0
                error('uavsim:launchFailed', 'Failed to launch MuJoCo simulator: %s', strtrim(command_output));
            end

            simulator_process_id = str2double(strtrim(command_output));
            if isnan(simulator_process_id)
                simulator_process_id = [];
            end

            uavsim.Launch.wait_for_simulator_startup(simulator_options);
        end

        function cleanup_simulator_process(simulator_process_id, simulator_options)
            if isempty(simulator_process_id) || ~simulator_options.shutdown_on_exit
                return;
            end

            if ispc
                kill_command = sprintf( ...
                    'powershell -NoProfile -Command "Stop-Process -Id %d -Force -ErrorAction SilentlyContinue"', ...
                    simulator_process_id ...
                );
            elseif isunix
                kill_command = sprintf( ...
                    'bash -lc "kill %d >/dev/null 2>&1 || true"', ...
                    simulator_process_id ...
                );
            else
                warning('uavsim:unsupportedPlatform', 'Simulator auto-shutdown is not implemented for this operating system.');
                return;
            end
            system(kill_command);
        end

        function exists_flag = does_udp_port_exist(local_port)
            if ispc
                command_text = sprintf( ...
                    'powershell -NoProfile -Command "if (Get-NetUDPEndpoint -LocalPort %d -ErrorAction SilentlyContinue) { Write-Output 1 } else { Write-Output 0 }"', ...
                    local_port);
            elseif isunix
                command_text = sprintf( ...
                    'bash -lc "if command -v ss >/dev/null 2>&1; then if ss -lun | awk ''{print \$5}'' | grep -Eq '':%d$''; then echo 1; else echo 0; fi; elif command -v netstat >/dev/null 2>&1; then if netstat -lun 2>/dev/null | awk ''{print \$4}'' | grep -Eq '':%d$''; then echo 1; else echo 0; fi; else echo 0; fi"', ...
                    local_port, local_port);
            else
                exists_flag = false;
                return;
            end
            [status, command_output] = system(command_text);
            exists_flag = status == 0 && strcmp(strtrim(command_output), '1');
        end

        function assert_udp_port_available(local_port, instance_id)
            if ~uavsim.Launch.does_udp_port_exist(local_port)
                return;
            end

            owner_summary = uavsim.Launch.describe_udp_port_owner(local_port);
            error('uavsim:portInUse', [ ...
                'UDP port %d is already in use before the controller socket could bind. ', ...
                'This usually means another MATLAB controller process is still running or a manual session owns the port. ', ...
                'Use a different instance_id, stop the other controller, or fully close the MATLAB session that owns the socket. ', ...
                'instance_id=%d expects controller_local_port=%d. %s' ...
            ], local_port, instance_id, local_port, owner_summary);
        end
    end

    methods (Static, Access = private)
        function wait_for_simulator_startup(simulator_options)
            deadline = tic;
            while toc(deadline) < simulator_options.wait_for_startup_seconds
                if uavsim.Launch.does_udp_port_exist(simulator_options.simulator_receive_port)
                    fprintf('MuJoCo simulator is ready on UDP port %d.\n', simulator_options.simulator_receive_port);
                    return;
                end
                pause(0.1);
            end

            fprintf('MuJoCo simulator launch was requested, but UDP port %d did not open within %.1f s.\n', ...
                simulator_options.simulator_receive_port, ...
                simulator_options.wait_for_startup_seconds ...
            );
        end

        function owner_summary = describe_udp_port_owner(local_port)
            owner_summary = 'Owner process could not be resolved.';
            if ispc
                command_text = sprintf([ ...
                    'powershell -NoProfile -Command "$endpoint = Get-NetUDPEndpoint -LocalPort %d -ErrorAction SilentlyContinue | Select-Object -First 1; ' ...
                    'if (-not $endpoint) { return }; ' ...
                    '$process = Get-Process -Id $endpoint.OwningProcess -ErrorAction SilentlyContinue; ' ...
                    'if ($process) { Write-Output (''Owner process: '' + $process.ProcessName + '' (PID '' + $process.Id + '')'') } else { Write-Output (''Owner PID: '' + $endpoint.OwningProcess) }"' ...
                ], local_port);
            elseif isunix
                command_text = sprintf( ...
                    'bash -lc "if command -v lsof >/dev/null 2>&1; then lsof -nP -iUDP:%d 2>/dev/null | tail -n +2 | head -n 1 | awk ''{print \"Owner process: \" $1 \" (PID \" $2 \")\"}''; fi"', ...
                    local_port);
            else
                return;
            end

            [status, command_output] = system(command_text);
            if status ~= 0
                return;
            end

            command_output = strtrim(command_output);
            if ~isempty(command_output)
                owner_summary = command_output;
            end
        end

        function python_executable = get_default_python_executable(simulator_root)
            if ispc
                python_executable = fullfile(simulator_root, '.venv', 'Scripts', 'python.exe');
                return;
            end

            python_executable = fullfile(simulator_root, '.venv', 'bin', 'python');
        end

        function [command_text, launch_mode] = build_simulator_launch_command(simulator_options)
            instance_id = round(simulator_options.instance_id);
            num_uavs = round(simulator_options.num_uavs);
            spawn_radius = simulator_options.spawn_radius;

            if ispc
                [command_text, launch_mode] = uavsim.Launch.build_windows_simulator_launch_command(simulator_options, instance_id, num_uavs, spawn_radius);
                return;
            end

            if isunix
                [command_text, launch_mode] = uavsim.Launch.build_unix_simulator_launch_command(simulator_options, instance_id, num_uavs, spawn_radius);
                return;
            end

            error('uavsim:unsupportedPlatform', 'Simulator auto-launch is not implemented for this operating system.');
        end

        function [command_text, launch_mode] = build_windows_simulator_launch_command(simulator_options, instance_id, num_uavs, spawn_radius)
            working_directory = uavsim.Launch.escape_powershell_string(simulator_options.working_directory);
            simulator_root = uavsim.Launch.escape_powershell_string(simulator_options.simulator_root);
            params_path = uavsim.Launch.escape_powershell_string(simulator_options.params_path);
            generated_xml_directory = uavsim.Launch.escape_powershell_string(simulator_options.generated_xml_directory);
            extra_arguments = uavsim.Launch.build_windows_cli_arguments(simulator_options);
            python_executable_path = uavsim.Launch.get_default_python_executable(simulator_options.simulator_root);

            if isfile(python_executable_path)
                python_executable = uavsim.Launch.escape_powershell_string(python_executable_path);
                command_text = sprintf( ...
                    'powershell -NoProfile -Command "$processArgs = @(''-m'',''wheeled_uav.cli'',''simulate'',''--instance-id'',''%d'',''--num-uavs'',''%d'',''--spawn-radius'',''%.9g'',''--params-file'',''%s'',''--generated-xml-dir'',''%s'') + %s; $process = Start-Process -FilePath ''%s'' -ArgumentList $processArgs -WorkingDirectory ''%s'' -PassThru; Write-Output $process.Id"', ...
                    instance_id, num_uavs, spawn_radius, params_path, generated_xml_directory, extra_arguments, python_executable, simulator_root);
                launch_mode = '.venv python -m wheeled_uav.cli';
                return;
            end

            command_text = sprintf( ...
                'powershell -NoProfile -Command "$processArgs = @(''run'',''--project'',''%s'',''mujoco-wheeled-uav-simulator'',''simulate'',''--instance-id'',''%d'',''--num-uavs'',''%d'',''--spawn-radius'',''%.9g'',''--params-file'',''%s'',''--generated-xml-dir'',''%s'') + %s; $process = Start-Process -FilePath ''uv'' -ArgumentList $processArgs -WorkingDirectory ''%s'' -PassThru; Write-Output $process.Id"', ...
                simulator_root, instance_id, num_uavs, spawn_radius, params_path, generated_xml_directory, extra_arguments, working_directory);
            launch_mode = 'uv run --project';
        end

        function [command_text, launch_mode] = build_unix_simulator_launch_command(simulator_options, instance_id, num_uavs, spawn_radius)
            working_directory = uavsim.Launch.escape_bash_double_quoted_string(simulator_options.working_directory);
            simulator_root = uavsim.Launch.escape_bash_double_quoted_string(simulator_options.simulator_root);
            params_path = uavsim.Launch.escape_bash_double_quoted_string(simulator_options.params_path);
            generated_xml_directory = uavsim.Launch.escape_bash_double_quoted_string(simulator_options.generated_xml_directory);
            extra_arguments = uavsim.Launch.build_unix_cli_arguments(simulator_options);
            python_executable_path = uavsim.Launch.get_default_python_executable(simulator_options.simulator_root);

            if isfile(python_executable_path)
                python_executable = uavsim.Launch.escape_bash_double_quoted_string(python_executable_path);
                command_text = sprintf( ...
                    'bash -lc "cd \"%s\"; nohup \"%s\" -m wheeled_uav.cli simulate --instance-id %d --num-uavs %d --spawn-radius %.9g --params-file \"%s\" --generated-xml-dir \"%s\"%s >/dev/null 2>&1 & echo $!"', ...
                    simulator_root, python_executable, instance_id, num_uavs, spawn_radius, params_path, generated_xml_directory, extra_arguments);
                launch_mode = '.venv python -m wheeled_uav.cli';
                return;
            end

            command_text = sprintf( ...
                'bash -lc "cd \"%s\"; nohup uv run --project \"%s\" mujoco-wheeled-uav-simulator simulate --instance-id %d --num-uavs %d --spawn-radius %.9g --params-file \"%s\" --generated-xml-dir \"%s\"%s >/dev/null 2>&1 & echo $!"', ...
                working_directory, simulator_root, instance_id, num_uavs, spawn_radius, params_path, generated_xml_directory, extra_arguments);
            launch_mode = 'uv run --project';
        end

        function cli_arguments = build_windows_cli_arguments(simulator_options)
            cli_arguments = '@()';
            argument_list = {};
            if uavsim.Util.get_struct_field(simulator_options, 'headless', false)
                argument_list{end + 1} = '''--headless''';
            end
            duration_seconds = uavsim.Util.get_struct_field(simulator_options, 'simulation_duration_seconds', NaN);
            if isfinite(duration_seconds)
                argument_list{end + 1} = '''--duration-seconds''';
                argument_list{end + 1} = sprintf('''%.9g''', double(duration_seconds));
            end
            record_path = char(uavsim.Util.get_struct_field(simulator_options, 'record_path', ''));
            if ~isempty(record_path)
                argument_list{end + 1} = '''--record''';
                argument_list{end + 1} = sprintf('''%s''', uavsim.Launch.escape_powershell_string(record_path));
            end
            if ~isempty(argument_list)
                cli_arguments = ['@(' strjoin(argument_list, ',') ')'];
            end
        end

        function cli_arguments = build_unix_cli_arguments(simulator_options)
            cli_arguments = '';
            if uavsim.Util.get_struct_field(simulator_options, 'headless', false)
                cli_arguments = [cli_arguments ' --headless'];
            end
            duration_seconds = uavsim.Util.get_struct_field(simulator_options, 'simulation_duration_seconds', NaN);
            if isfinite(duration_seconds)
                cli_arguments = sprintf('%s --duration-seconds %.9g', cli_arguments, double(duration_seconds));
            end
            record_path = char(uavsim.Util.get_struct_field(simulator_options, 'record_path', ''));
            if ~isempty(record_path)
                cli_arguments = sprintf('%s --record \"%s\"', cli_arguments, uavsim.Launch.escape_bash_double_quoted_string(record_path));
            end
        end

        function escaped_text = escape_powershell_string(text)
            escaped_text = strrep(text, '''', '''''');
        end

        function escaped_text = escape_bash_double_quoted_string(text)
            escaped_text = char(text);
            escaped_text = strrep(escaped_text, '\', '\\');
            escaped_text = strrep(escaped_text, '"', '\"');
            escaped_text = strrep(escaped_text, '$', '\$');
            escaped_text = strrep(escaped_text, '`', '\`');
        end
    end
end
