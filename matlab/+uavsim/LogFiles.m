classdef LogFiles
    % UAVSIM.LOGFILES  Shared .mat log discovery/loading for review scripts.
    methods (Static)
        function log_path = resolve_log_path(project_directory, candidate_path)
            % '' -> newest log in <project>/logs; otherwise absolute, CWD-relative,
            % or project-relative path (first match wins).
            if nargin < 2 || isempty(candidate_path)
                log_path = uavsim.LogFiles.newest_log(project_directory, '*.mat');
                fprintf('Using latest log: %s\n', log_path);
                return;
            end

            candidate_path = char(candidate_path);
            if isfile(candidate_path)
                log_path = candidate_path;
                return;
            end

            project_relative_path = fullfile(project_directory, candidate_path);
            if isfile(project_relative_path)
                log_path = project_relative_path;
                return;
            end

            error('uavsim:logNotFound', 'Log file not found: %s', candidate_path);
        end

        function log_path = newest_log(project_directory, file_pattern)
            logs_directory = fullfile(project_directory, 'logs');
            listing = dir(fullfile(logs_directory, file_pattern));
            % Formation bundles hold a `formation_log` variable, not `log`,
            % and (with the default bundle_only mode) are always the newest
            % .mat after a formation run — skip them so single-log consumers
            % keep finding the newest per-run `log` file. Bundle-aware tools
            % do their own discovery.
            bundle_mask = startsWith({listing.name}, 'formation_bundle');
            listing = listing(~bundle_mask);
            if isempty(listing)
                error('uavsim:noLogs', 'No log files matching %s under %s. Run a controller first.', file_pattern, logs_directory);
            end
            [~, newest_index] = max([listing.datenum]);
            log_path = fullfile(logs_directory, listing(newest_index).name);
        end

        function log = load_log(log_path)
            loaded = load(log_path);
            if ~isfield(loaded, 'log')
                error('uavsim:invalidLog', 'The file does not contain a log variable: %s', log_path);
            end
            log = loaded.log;
        end

        function timestamp = extract_timestamp(file_path)
            % 'yyyyMMdd_HHmmss' token right before .mat, or '' when absent.
            timestamp = '';
            token = regexp(char(file_path), '\d{8}_\d{6}(?=\.mat$)', 'match', 'once');
            if ~isempty(token)
                timestamp = token;
            end
        end
    end
end
