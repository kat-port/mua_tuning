
mice_to_plot = [3 5 11];
plotWindow = [-0.1, 0.2];
save_plot = 'true';
stimulusOnset = [0];
for iM = 1:length(mice_to_plot)
    % Load data
    mouse_row = mice_to_plot(iM);
    date = experiment_info(mouse_row).date;
    mouse = experiment_info(mouse_row).mouse;
    path_to_sweep = fullfile(experiment_info(mouse_row).path,'sweeps');
    coi = experiment_info(mouse_row).l4_chan_csd;
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
    n_trials = 1000;
    all_stim_relative_spike_times = cell(n_trials,length(vars_to_plot));
    trial_spike_times = cell(n_trials,1);
    trial_relative_spike_times = cell(n_trials,1);
    spike_times_this_channel = mua_spike_data.all_spike_times(mua_spike_data.chan_id == coi); %find spike times for these
    
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
                all_stim_relative_spike_times{iT,iV} = spikes_relative_onset;
            end
        end

    end

    figure
    for iV = 1:length(vars_to_plot)
        spikes_to_plot = all_stim_relative_spike_times(:,iV);
        % Create scatter plot
        s = subplot(2,1,iV);
        for i = 1:n_trials
        if ~isempty(cell2mat(spikes_to_plot(i)))
            plot(cell2mat(spikes_to_plot(i)),i*ones(length(spikes_to_plot(i)),1),'k.', 'MarkerSize', 1)   
        end
        hold on
        end
        if strcmp(vars_to_plot(iV),fam)
            label = 'Familiar Sweep';
        elseif strcmp(vars_to_plot(iV),nov)
            label = 'Novel Sweep';
        end
        x_coord = ([0 0.1 0.1 0]);
        y_coord = ([0 0 n_trials n_trials]);
        fill(x_coord,y_coord,[0.5 0.5 0.5],'EdgeColor','none','FaceAlpha',0.3)
        xlim(plotWindow)
        if iV == 1
            s.XTickLabel={};
        end
        title([label,' ','(',vars_to_plot{iV},')'])
        raster_plot_title = [' Raster Channel: ',num2str(coi)];
        sgtitle([mouse,raster_plot_title])
    end
    if strcmp(save_plot,'true')
    set(gcf,'Position', get(0, 'Screensize'))
    saveas(gcf,[mouse,'_fam_v_nov_l4_approx_raster.fig'])
    saveas(gcf,[mouse,'_fam_v_nov_l4_approx_raster.png'])
    print(gcf,[mouse,'_fam_v_nov_l4_approx_raster'],'-dpdf','-fillpage')
    end
end

