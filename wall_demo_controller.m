function wall_demo_controller(varargin)
% WALL_DEMO_CONTROLLER  Minimal wall-riding demo (single UAV).
%
% Sample entry point kept at the repository root for quick use. Project
% repositories should keep their own controllers and use this as a reference.
%
% A deliberately trivial baseline, not a research controller: the UNMODIFIED
% shared hover controller is given a target position BEHIND the wall face,
% so the wall stops the vehicle and the position error provides the press
% while the target's height follows a scripted climb/hold/descend profile.
%
% Options: 'press_depth_m', 'z_low', 'z_high', 'contact_x' plus the shared
% uavsim.RunOptions set (instance_id, duration_seconds, auto_launch,
% state_timeout_seconds, headless, ...).
project_directory = fileparts(mfilename('fullpath'));
matlab_directory = fullfile(project_directory, 'matlab');
implementation_directory = fullfile(matlab_directory, 'controllers');
shared_directory = fullfile(matlab_directory, 'shared');
addpath(matlab_directory, implementation_directory, shared_directory);
cleanup_handler = onCleanup(@() rmpath(matlab_directory, implementation_directory, shared_directory));
wall_demo_controller_impl(varargin{:});
end
