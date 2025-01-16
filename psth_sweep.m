%final_psth
mice_to_plot = [11];
save_plot = 'true';
plotWindow = [-0.1, 0.2];
baselineWindow = [-0.1 0];

chans = [23:35];
stimulusOnset = [0];
bin_width_psth = 1/100;
edges_psth = plotWindow(1):bin_width_psth:plotWindow(2);
binCenters = edges_psth(1:end-1) + bin_width_psth/2;  % Compute bin centers for plotting
baseline_bin_idx = find(binCenters >= baselineWindow(1) & binCenters < baselineWindow(2)); % Find bins that correspond to baseline        idx_dur = find(binCenters >= durWindow(1) & binCenters < durWindow(2));
n_bins_psth = floor((plotWindow(2)-plotWindow(1))/bin_width_psth);

for iM = 1:length(mice_to_plot)
    % Load data
    mouse_row = mice_to_plot(iM);
    date = experiment_info(mouse_row).date;
    mouse = experiment_info(mouse_row).mouse;
    coi = experiment_info(mouse_row).l4_chan_csd;
    path_to_sweep = fullfile(experiment_info(mouse_row).path,'sweeps');
    cd(path_to_sweep)
    load([mouse,'_srp_sweep_mua_spike_data.mat']);
    load([mouse,'_srp_sweep_cleaned_played_stim.mat']);

    % Get stim labels
    fam = experiment_info(mouse_row).fam_id;
    if strcmp(fam,'up')
        nov = 'down';
    elseif strcmp(fam,'down')
        nov = 'up';
    end
    var1_to_plot = fam;
    var2_to_plot = nov;
    vars_to_plot = {fam ,nov};
    all_stim_labels = clean_played_stim.stimulus_name;


    %%
    % initialise empty variables
    n_trials = 1000;
    all_stim_relative_spike_times = cell(n_trials,length(vars_to_plot));
    all_stim_fr = zeros(n_trials,n_bins_psth,length(vars_to_plot));
    trial_spike_times = cell(n_trials,1);
    trial_relative_spike_times = cell(n_trials,1);
    trial_spike_count = zeros(n_trials,n_bins_psth);
    trial_fr = zeros(n_trials,n_bins_psth);
    % get spikes
    spike_times_this_channel = mua_spike_data.all_spike_times(mua_spike_data.chan_id == coi); %find spike times for these

    figure
    for iV = 1:length(vars_to_plot)
        trial_type = vars_to_plot{iV};
        onset_times = clean_played_stim.onset_time(strcmp(clean_played_stim.stimulus_name,trial_type));
        sound_on = unique(onset_times);

        for iT = 1:length(sound_on)
            one_trial_spikes = spike_times_this_channel(spike_times_this_channel >= sound_on(iT)+plotWindow(1) & spike_times_this_channel <= sound_on(iT) + plotWindow(2)); %find the clock spike times that fall between sound onset and offset (+extra window)
            spikes_relative_onset = one_trial_spikes - sound_on(iT);
            if isempty(spikes_relative_onset)
                trial_spike_times{iT} = [];
                trial_relative_spike_times{iT} = [];
            else
                spike_count = histcounts(spikes_relative_onset, edges_psth);
                one_trial_fr = spike_count/bin_width_psth;
                trial_spike_times{iT} = one_trial_spikes;
                trial_relative_spike_times{iT} = spikes_relative_onset;
                trial_spike_count(iT,:) = spike_count;
                trial_fr(iT,:) = one_trial_fr;
                all_stim_relative_spike_times{iT,iV} = spikes_relative_onset;
            end
        end
        psth = mean(trial_fr,1);
        mean_baseline = mean(psth(baseline_bin_idx)); % Compute mean baseline fr
        sd_baseline = std(psth(baseline_bin_idx));
        psth_baseline_norm = (psth-mean_baseline)./sd_baseline;
        %figure
        plot(binCenters, psth_baseline_norm)
        hold on
    end
    x_coord = ([0 0.1 0.1 0]);
    y_coord = ([min(ylim) min(ylim) max(ylim) max(ylim)]);
    fill(x_coord,y_coord,[0.5 0.5 0.5],'EdgeColor','none','FaceAlpha',0.3)
    xlim([plotWindow(1) plotWindow(2)])
    set(gca, 'XTick', [-0.1 0 0.1 0.2], 'XTickLabel', {'-0.1', '0', '0.1', '0.2'});
    title([mouse, ' Familiar vs Novel Z-Scored PSTH'])
    subtitle(['Channel: ',num2str(coi), ' Bin Width: ', num2str(bin_width_psth)])
    legend(['Familiar ','(',vars_to_plot{1},')'],['Novel ','(',vars_to_plot{2},')'])
    if strcmp(save_plot,'true')
    set(gcf,'Position', get(0, 'Screensize'))
    saveas(gcf,[mouse,'_fam_v_nov_l4_approx_psth.fig'])
    saveas(gcf,[mouse,'_fam_v_nov_l4_approx_psth.png'])
    print(gcf,[mouse,'_fam_v_nov_l4_approx_psth'],'-dpdf','-fillpage')
    end
end