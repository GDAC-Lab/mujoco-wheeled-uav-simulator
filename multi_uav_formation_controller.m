function multi_uav_formation_controller(varargin)
% MULTI_UAV_FORMATION_CONTROLLER  Circular formation control for multiple UAVs in one world.
%
% Sample entry point kept at the repository root for quick use. Project
% repositories should keep their own controllers and use this as a reference.
%
% Options: 'num_uavs', 'formation_radius', 'centroid_target_xy',
% 'centroid_gain', 'formation_gain', 'base_height', 'formation_log_mode',
% gain vectors, plus the shared uavsim.RunOptions set. See
% matlab/controllers/multi_uav_formation_controller_impl.m for the full list.
project_directory = fileparts(mfilename('fullpath'));
matlab_directory = fullfile(project_directory, 'matlab');
implementation_directory = fullfile(matlab_directory, 'controllers');
shared_directory = fullfile(matlab_directory, 'shared');
addpath(matlab_directory, implementation_directory, shared_directory);
cleanup_handler = onCleanup(@() rmpath(matlab_directory, implementation_directory, shared_directory));
multi_uav_formation_controller_impl(varargin{:});
end
