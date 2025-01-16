%% LFP
BaselineTime = [0 0.1];
coi = experiment_info(5).l4_chan_csd;
SR = experiment_info(5).SR;
mouse = experiment_info(5).mouse;
lfp = sweep_selec_raw.lfp_raw;
stim_info = sweep_selec_raw.stim_ids;
unique_durations = unique(stim_info.duration);
unique_directions = unique(stim_info.direction);

figure; % Create a new figure

% Loop through durations and directions
for dir = 1:length(unique_directions)
    for dur = 1:length(unique_durations)
        % Get the LFP mean for the current duration and direction
        [lfp_mean] = get_mean_lfp_sweep_selec(sweep_selec_raw, unique_durations(dur), unique_directions{dir}, coi, SR, BaselineTime, 'normalise');

        % Create a time vector
        time = linspace(-0.2, 0.3, size(lfp_mean, 2));

        % Calculate subplot index (9 rows, 2 columns)
        subplot_idx = (dur - 1) * length(unique_directions) + dir;

        % Create subplot
        subplot(length(unique_durations), 2, subplot_idx);

        % Plot the data
        plot(time, lfp_mean);

        % Add title and labels
        title(sprintf('Dur: %d, Dir: %s', round(unique_durations(dur),2), unique_directions{dir}));
        xlabel('Time (s)');
        ylabel('LFP Mean');
    end
end

% Add a super title for the figure
sgtitle('LFP Mean for Each Duration and Direction');

%% MUA
baselineWindow = [-0.1 0];
plotWindow = [-0.1, 0.1];
bin_width_psth = 1/80;

%% IF PLOTTING ALL CHANNELS
chans = [10 20 30 40 50 60];
fig = figure;
allYLim = [];
for iC = 1:length(chans)
    spike_times_this_channel = mua_spike_data.all_spike_times(mua_spike_data.chan_id == chans(iC));
       for dir = 1:length(unique_directions)
            durWindow = [0 0.1];
            [all_stim_fr] = get_raw_fr_sweep_dir_selec(unique_directions(dir),unique_durations,clean_played_stim,spike_times_this_channel,plotWindow,bin_width_psth);
            [~,peak_fr_all_stim] = peak_fr_general(all_stim_fr,unique_durations,bin_width_psth,plotWindow,baselineWindow,durWindow);
            s = subplot(length(unique_directions), length(chans), (dir - 1) * length(chans) + iC);
            plot(peak_fr_all_stim)
            title(unique_directions(dir))
            allYLim = [allYLim; ylim];

            if dir == length(unique_directions)
                s.XTickLabel= {num2str(round(unique_durations,2))};
            else
                xlabel('');
                s.XTick = {};
                s.XTickLabel= {};
            end

       end
end
minY = min(allYLim(:,1));
maxY = max(allYLim(:,2));
for iY = 1:length(allYLim)
    subplot(length(unique_directions),length(chans),iY)
    ylim([minY maxY]);
end
sgtitle(mouse)
%% IF PLOTTING ONE CHANNEL
mice_to_plot = [3 4 5 9 11];
for iM = 1:length(mice_to_plot)

    mouse_row = mice_to_plot(iM);
    date = experiment_info(mouse_row).date;
    mouse = experiment_info(mouse_row).mouse;
    path_to_sweep_selec = fullfile(experiment_info(mouse_row).path,'sweep_selec');
    cd(path_to_sweep_selec)
    load([mouse,'_sweep_selec_mua_spike_data.mat']);
    load([mouse,'_sweep_selec_cleaned_played_stim.mat']);
    coi = experiment_info(mouse_row).l4_chan_csd;
    SR = experiment_info(mouse_row).SR;
    unique_durations = unique(mua_spike_data.sweep_duration);
    unique_durations = unique_durations(unique_durations ~= 0);
    directions = mua_spike_data.sweep_direction;
    for i = 1:length(directions)
        if isempty(directions{i})
            directions{i} = 'empty';
        end
    end

    unique_directions = unique(directions);
    unique_directions = setdiff(unique_directions, {'empty'});

    figure;

    spike_times_this_channel = mua_spike_data.all_spike_times(mua_spike_data.chan_id == coi);
    for dir = 1:length(unique_directions)
        
        [all_stim_fr] = get_raw_fr_sweep_dir_selec(unique_directions(dir),unique_durations,clean_played_stim,spike_times_this_channel,plotWindow,bin_width_psth);
        
        [~,peak_fr_all_stim] = peak_fr_sweep_selec(all_stim_fr,unique_durations,bin_width_psth,plotWindow,baselineWindow);
        plot(peak_fr_all_stim)
        hold on
    end

    legend(unique_directions{1},unique_directions{2})
    xticklabels({num2str(round(unique_durations,2))});
    title([mouse, ' Sweep Direction Tuning'])
    subtitle(['Chan ', num2str(coi)])
    box off
    xlabel('Duration (s)')
    ylabel('Peak FR (sp/s)')
end