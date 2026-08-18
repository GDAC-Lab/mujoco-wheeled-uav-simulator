classdef Metrics
    % UAVSIM.METRICS  Controller-side runtime metrics for logging/diagnostics.
    methods (Static)
        function runtime_metrics = initialize()
            runtime_metrics = struct( ...
                'last_state_sequence', NaN, ...
                'last_state_age_ms', NaN, ...
                'last_state_sequence_gap', 0.0, ...
                'state_sequence_gap_count', 0.0, ...
                'timeout_count', 0.0, ...
                'last_controller_compute_ms', NaN, ...
                'command_sequence', 0.0, ...
                'last_source_state_sequence', NaN ...
            );
        end

        function runtime_metrics = update(runtime_metrics, state, controller_compute_ms)
            state_metrics = uavsim.Protocol.get_state_packet_metrics(state);
            previous_sequence = runtime_metrics.last_state_sequence;
            current_sequence = state_metrics.sequence;

            if ~isnan(previous_sequence) && ~isnan(current_sequence)
                sequence_gap = max(0.0, current_sequence - previous_sequence - 1.0);
            else
                sequence_gap = 0.0;
            end

            runtime_metrics.last_state_sequence = current_sequence;
            runtime_metrics.last_state_age_ms = state_metrics.age_ms;
            runtime_metrics.last_state_sequence_gap = sequence_gap;
            runtime_metrics.state_sequence_gap_count = runtime_metrics.state_sequence_gap_count + sequence_gap;
            runtime_metrics.last_controller_compute_ms = double(controller_compute_ms);
            runtime_metrics.command_sequence = runtime_metrics.command_sequence + 1.0;
            runtime_metrics.last_source_state_sequence = current_sequence;
        end

        function runtime_metrics = note_timeout(runtime_metrics)
            runtime_metrics.timeout_count = runtime_metrics.timeout_count + 1.0;
        end
    end
end
