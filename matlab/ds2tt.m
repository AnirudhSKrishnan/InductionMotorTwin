function TT = ds2tt(ds)
%DS2TT  Convert Simulink.SimulationData.Dataset -> timetable
%   TT = ds2tt(ds)

    if ~isa(ds,'Simulink.SimulationData.Dataset')
        error('ds2tt:Input', 'Input must be Simulink.SimulationData.Dataset, got %s.', class(ds));
    end

    TT = timetable();  % start empty

    % FIX: Use numElements(ds) method instead of .NumElements property
    num_sigs = numElements(ds);

    for k = 1:num_sigs
        % Use {} indexing to get the element
        element = ds{k};
        
        % 1. Safety Check: Ensure the element is a Signal with Timeseries
        % Some elements might be empty or just metadata
        if ~isa(element, 'Simulink.SimulationData.Signal')
             % Try to handle it if it's just a raw timeseries inside the dataset
             if isa(element, 'timeseries')
                 ts = element;
                 baseName = ts.Name;
             else
                 warning('Skipping element %d: Not a Signal or Timeseries.', k);
                 continue;
             end
        else
             % Standard Case: Signal Object
             ts = element.Values;
             baseName = element.Name;
        end
        
        if isempty(ts)
            continue;
        end

        t = ts.Time;
        x = ts.Data;

        % 2. Handle 3D data (Simulink sometimes outputs 1x1xN)
        x = squeeze(x); 
        
        % Ensure data is oriented correctly (Time should be rows)
        if size(x, 1) ~= length(t)
             x = x'; 
        end

        % 3. Naming Logic
        if isempty(baseName)
            baseName = sprintf('sig%d',k);
        end
        % Clean invalid characters from names
        baseName = matlab.lang.makeValidName(baseName);

        % 4. Create Sub-Timetable
        numCols = size(x, 2);
        if numCols == 1
            varNames = {baseName};
        else
            varNames = arrayfun(@(i) sprintf('%s_%d', baseName, i), ...
                                1:numCols, 'UniformOutput', false);
        end
        
        TTk = array2timetable(x, 'RowTimes', seconds(t), 'VariableNames', varNames);

        % 5. Merge Data
        if isempty(TT)
            TT = TTk;
        else
            % Use 'union' to keep all data points from different sample times
            TT = synchronize(TT, TTk, 'union', 'linear');
        end
    end
end