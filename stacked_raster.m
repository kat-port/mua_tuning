mice_to_plot = [3]; % 2 6 7 8 10]; %[3 4 5 9 11];
for iM = 1:length(mice_to_plot)
    mouse_row = mice_to_plot(iM);
    date = experiment_info(mouse_row).date;
    mouse = experiment_info(mouse_row).mouse;
    path_to_pt = fullfile(experiment_info(mouse_row).path,'pt');
    cd(path_to_pt)
    load([mouse,'_pt_trial_table.mat']);
    all_freq = unique(trial_table.freq);
    frequencies = all_freq(1:11);
    n_freqs = length(frequencies);
    atten = 30;
    chans = [20];
    filename = '_pt_stacked_raster_mua_chan_';
    plot_window = [-0.2 0.3];
    plot_freqs = [2 4 8 16 32 64];
    
    for iC = 1:length(chans)
        channel = chans(iC);

        for iF = 1:n_freqs
            % Clear all_trial_spikes for each new frequency
            all_trial_spikes = {};

            % Find the trials that match the current frequency and attenuation
            stim_idx = trial_table.freq == frequencies(iF) & trial_table.atten == atten;
            MUA_cell_array = trial_table.MUAData(stim_idx);

            % Loop over trials and extract spike times for the specific channel
            for iT = 1:length(MUA_cell_array)
                trial_spikes = MUA_cell_array{iT}{channel, 1};
                all_trial_spikes{iT} = trial_spikes;
            end

            % Store spike times for each frequency
            spike_times{iF,1} = all_trial_spikes;
        end
        

  
       figure
       
        % Loop over each frequency and create a subplot
        for i = 1:n_freqs 
            stim_spikes = spike_times{i}; %index the cell containing spike times for trials when that frequency was played
            n_trials = length(stim_spikes); %number of trials = number of cells 
            subplot(n_freqs, 1, i); % Create stacked subplots
            hold on;
           
            % Plot raster for each trial
            for trial = 1:n_trials 
                trial_spikes = stim_spikes{trial};
                trial_spikes_windowed = trial_spikes(trial_spikes >= plot_window(1) & trial_spikes <= plot_window(2));
                y = trial * ones(1, length(trial_spikes_windowed));
                plot(trial_spikes_windowed, y, 'k.', 'MarkerSize', 10);% Plot spikes as black dots
            end
            
            xlim([plot_window(1) plot_window(2)])
            % Formatting
            ylim([0 n_trials+1]);

            if ismember(frequencies(i), plot_freqs)
                ylabel(num2str(round(frequencies(i),2)))
                yticklabels([])
            else
                set(gca, 'YTickLabel', []); % No label for other frequencies
            end
            if i == n_freqs
                xlabel('Time (s)');
                set(gca,'TickDir','none', 'XTick', [-0.1 0 0.1 0.2], 'XTickLabel', {'-0.1', '0', '0.1', '0.2'});
            else
                set(gca,'TickDir','none', 'XTickLabel', []); % Remove x-tick labels for the first plots
            end

            % Add grid lines and dashed vertical lines

            xline(0, '--k', 'LineWidth', 1.5); % Dashed vertical line at 0 seconds
            xline(0.1, '--k', 'LineWidth', 1.5); % Dashed vertical line at 0.1 seconds
        end

        % Add a title to the top plot
        subplot(n_freqs, 1, 1);
        title(['Mouse ',mouse,' Raster Plot for Each Frequency - Stacked. Chan:', num2str(chans(iC))]);
        hold off
%        set(gcf, 'Position', get(0, 'Screensize'));
%         saveas(gcf,[mouse,filename,num2str(chans(iC)),'.fig'])
%         saveas(gcf,[mouse,filename,num2str(chans(iC)),'.png'])
%        print(gcf,[mouse,filename,num2str(chans(iC))],'-dpdf','-fillpage')
    end

end