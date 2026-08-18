classdef Joystick
    % UAVSIM.JOYSTICK  HID joystick access for manual flight modes.
    %
    % Uses sim3d.io.Joystick (Simulink 3D Animation) when available and falls
    % back to the legacy vrjoystick interface. Both expose
    %   [axes, buttons, povs] = read(device)
    % All axis handling (deadzone, axis index selection, sign) lives here so
    % controllers only deal in physical commands.
    methods (Static)
        function joy_handle = open()
            joy_handle = [];
            if exist('sim3d.io.Joystick', 'class')
                try
                    joy_handle = sim3d.io.Joystick();
                catch
                end
            end
            if isempty(joy_handle) && exist('vrjoystick', 'file')
                try
                    % Deliberate legacy fallback for releases without
                    % sim3d.io.Joystick (which is tried first above).
                    joy_handle = vrjoystick(1); %#ok<VRJOYSTK>
                catch
                end
            end
            if isempty(joy_handle)
                return;
            end
            % Warm up HID (some pads return empty axes until initial reads).
            for warmup_index = 1:40
                [axes_data, axes_ok] = uavsim.Joystick.read_axes(joy_handle);
                if axes_ok && numel(axes_data) >= 2
                    break;
                end
                pause(0.025);
            end
        end

        function [axes_out, read_ok] = read_axes(joy_handle)
            axes_out = [];
            read_ok = false;
            if isempty(joy_handle)
                return;
            end
            try
                [axes_row, ~, ~] = read(joy_handle);
                if isempty(axes_row)
                    return;
                end
                axes_out = double(axes_row(:));
                read_ok = true;
            catch
            end
        end

        function value = axis_value(axes_data, axis_index, deadzone, invert_axis)
            % Deadzone-filtered value of one stick axis in [-1, 1].
            % Falls back to the last available axis when the pad has fewer
            % axes than the configured index (mirrors common 2-axis pads).
            persistent warned_axis_indices
            if isempty(axes_data)
                value = 0.0;
                return;
            end
            clamped_index = max(1, min(round(double(axis_index)), numel(axes_data)));
            if clamped_index ~= round(double(axis_index)) && ~ismember(round(double(axis_index)), warned_axis_indices)
                % Warn once so a 0-based index copied from HID docs (or a pad
                % with fewer axes) does not silently steer the wrong axis.
                warned_axis_indices(end + 1) = round(double(axis_index));
                warning('uavsim:axisIndexClamped', ...
                    'Joystick axis index %d is outside 1..%d; using axis %d. Axis indices are 1-based.', ...
                    round(double(axis_index)), numel(axes_data), clamped_index);
            end
            value = double(axes_data(clamped_index));
            if abs(value) < double(deadzone)
                value = 0.0;
            end
            if invert_axis
                value = -value;
            end
            value = max(-1.0, min(1.0, value));
        end
    end
end
