function hovering_controller(varargin)
% HOVERING_CONTROLLER  Geometric hover control sample (single UAV).
%
% Sample entry point kept at the repository root for quick use. Project
% repositories should keep their own controllers and use this as a reference.
%
% Options: 'target_position', [x; y; z] plus the shared uavsim.RunOptions set
% (instance_id, duration_seconds, auto_launch, state_timeout_seconds, ...).
project_directory = fileparts(mfilename('fullpath'));
matlab_directory = fullfile(project_directory, 'matlab');
implementation_directory = fullfile(matlab_directory, 'controllers');
shared_directory = fullfile(matlab_directory, 'shared');
addpath(matlab_directory, implementation_directory, shared_directory);
cleanup_handler = onCleanup(@() rmpath(matlab_directory, implementation_directory, shared_directory));
hovering_controller_impl(varargin{:});
end
