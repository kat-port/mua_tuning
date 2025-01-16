%% TO CHECK: NORM
octave_freqs = [2 4 8 16 32 64];
baselineWindow = [-0.1 0];
plotWindow = [-0.2, 0.3];
durWindow = [0 0.05];
bin_width_psth = 1/80;
chans = [10 20 30 40 50];
mice_to_plot = [10];
save_plot = 'false';
for iM = 1:length(mice_to_plot)


    mouse_row = mice_to_plot(iM);
    date = experiment_info(mouse_row).date;
    mouse = experiment_info(mouse_row).mouse;
    path_to_pt = fullfile(experiment_info(mouse_row).path,'pt');
    cd(path_to_pt)
    load([mouse,'_mua_spike_data.mat']);
    load([mouse,'_cleaned_played_stim.mat']);
    attens = unique(clean_played_stim.attens);
    freqs = unique(clean_played_stim.freqs);
    oct_freq_idx = find(ismember(freqs, octave_freqs));
    allYLim = [];
    fig = figure;
    for iC = 1:length(chans)
        spike_times_this_channel = mua_spike_data.all_spike_times(mua_spike_data.chan_id == chans(iC));
        for i = 1:length(attens)
            [all_stim_fr] = get_raw_fr(attens(i),freqs,clean_played_stim,spike_times_this_channel,plotWindow,bin_width_psth);
            [~,peak_fr_all_stim] = peak_fr_tuning(all_stim_fr,freqs,bin_width_psth,plotWindow,baselineWindow,durWindow);
            s = subplot(length(attens), length(chans), (i - 1) * length(chans) + iC);
            plot(peak_fr_all_stim)
            allYLim = [allYLim; ylim];
            if i == 1
                title(['Tuning for Channel ',num2str(chans(iC))])
            end

            if iC == 1
                ylabel(num2str(90-attens(i)))
            end

            if i == length(attens)

                s.XTick = oct_freq_idx;
                s.XTickLabel= {num2str(octave_freqs(1)),num2str(octave_freqs(2)),num2str(octave_freqs(3)),num2str(octave_freqs(4)),num2str(octave_freqs(5)),num2str(octave_freqs(6))};
            else
                xlabel('');
                s.XTick = {};
                s.XTickLabel= {};
            end

        end

    end
    %% Apply unified y-limits
    minY = min(allYLim(:,1));
    maxY = max(allYLim(:,2));
    for iY = 1:length(allYLim)
        subplot(length(attens),length(chans),iY)
        ylim([minY maxY]);
    end
    han=axes(fig,'visible','off');
    han.YLabel.Visible='on';
    han.XLabel.Visible='on';
    ylabel(han,'Sound Level (dB SPL)');
    xlabel(han,'Frequency (kHz)');
    sgtitle(mouse)
    if strcmp(save_plot,'true')
        set(gcf,'Position', get(0, 'Screensize'))
        saveas(gcf,[mouse,'_pt_psth_freq_intensity_by_chan.fig'])
        saveas(gcf,[mouse,'_pt_psth_freq_intensity_by_chan.png'])
        print(gcf,[mouse,'_pt_psth_freq_intensity_by_chan'],'-dpdf','-fillpage')
    end
end