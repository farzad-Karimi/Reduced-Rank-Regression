import numpy as np
import itertools
from scipy.stats import sem
import reduced_rank_regressor

def sqerr(matrix1, matrix2):
    """Squared error (frobenius norm of diff) between two matrices."""
    return np.corrcoef(matrix1.flatten(), matrix2.flatten())[0][1]**2

def detect_bursts(spike_train, burst_threshold):
    burst_indices = []
    current_burst = []
    for i in range(len(spike_train) - 1):
        if spike_train[i + 1] - spike_train[i] <= burst_threshold:
            current_burst.append(spike_train[i])
        else:
            if len(current_burst) > 0:
                burst_indices.append(current_burst)
            current_burst = []
    return burst_indices

def population_extractor_lick(cache, animal, area):
    session = animal
    try:
        Session = cache.get_ecephys_session(ecephys_session_id=session)
    except:
        return [], []
    units         = Session.get_units()
    channels      = Session.get_channels()
    unit_channels = units.merge(channels, left_on='peak_channel_id', right_index=True).sort_values('probe_vertical_position', ascending=False)
    good_unit_filter = ((unit_channels['snr']>1)&
                        (unit_channels['isi_violations']<1)&
                        (unit_channels['firing_rate']>0.1))
    good_units = unit_channels.loc[good_unit_filter] 
    good_units = good_units[good_units['structure_acronym']==area]
    ids = []
    for undex, _ in good_units.iterrows():
        ids.append(undex)
    all_spike_times = Session.spike_times
    first_lick = []
    for reward in Session.rewards[Session.rewards['auto_rewarded']==False]['timestamps']:
        first_lick.append(min(Session.licks['timestamps'], key=lambda x: abs(x - reward)))
    if len(first_lick) > 20:
        lick_pre, lick_post = [], []
        for neuron in ids:
            spike_times = all_spike_times[neuron]
            nburst      = 0
            for time in first_lick:          # burst detection loop
                stim_on = (spike_times[(spike_times>=time-.25) & (spike_times<time+.25)] - time+.25)*1000
                bursts  = detect_bursts(stim_on, burst_threshold=10)
                for burst in bursts:
                    if burst[0] < 100:
                        nburst += 1
                        break
            if nburst == 0:              # neurons with no burst trials are filtered
                continue
            post_lick, pre_lick = [], []
            for time in first_lick:
                post_lick_hist = np.histogram(spike_times[(spike_times>=time) & (spike_times<time+.25)] - time    , np.arange(0, .26, .025))[0]
                pre_lick_hist  = np.histogram(spike_times[(spike_times>time-.25) & (spike_times<=time)] - time+.25, np.arange(0, .26, .025))[0]
                post_lick.append(post_lick_hist)
                pre_lick.append(pre_lick_hist)
            lick_post.append(list(itertools.chain.from_iterable(post_lick)))
            lick_pre.append(list(itertools.chain.from_iterable(pre_lick)))
            
        lick_post = np.array(lick_post)
        lick_pre  = np.array(lick_pre)
        return lick_pre, lick_post
    else:
        return [], []

def population_extractor(cache, animal, area, image, datatype):
    session = animal
    try:
        Session = cache.get_ecephys_session(ecephys_session_id=session)
    except:
        return [], [], []
    units         = Session.get_units()
    channels      = Session.get_channels()
    unit_channels = units.merge(channels, left_on='peak_channel_id', right_index=True).sort_values('probe_vertical_position', ascending=False)

    good_unit_filter = ((unit_channels['snr']>1)&
                        (unit_channels['isi_violations']<1)&
                        (unit_channels['firing_rate']>0.1))
    good_units = unit_channels.loc[good_unit_filter] 
    good_units = good_units[good_units['structure_acronym']==area]
    ids = []
    for undex, _ in good_units.iterrows():
        ids.append(undex)

    all_spike_times        = Session.spike_times
    stimulus_presentations = Session.stimulus_presentations
    stim_table             = stimulus_presentations[(stimulus_presentations['stimulus_block']==0) & (stimulus_presentations['image_name']==image)]  # active block

    stims = stim_table['start_time'].values

    burst_threshold = 10
    
    Area_source, Area_target, Area = [], [], []
    for neuron in ids:
        spike_times = all_spike_times[neuron]
        nburst      = 0
        for time in stims:          # burst detection loop
            stim_on = (spike_times[(spike_times>=time) & (spike_times<time+.25)] - time)*1000
            bursts  = detect_bursts(stim_on, burst_threshold)
            for burst in bursts:
                if burst[0] < 100:
                    nburst += 1
                    break
        if nburst == 0:              # neurons with no burst trials are filtered
            continue
        a_source, a_target, a = [], [], []
        for time in stims:
            trial_hist = np.histogram(spike_times[(spike_times>=time) & (spike_times<time+.25)] - time, np.arange(0, .26, .025))[0]
            a_source.append(trial_hist[:-1])
            a_target.append(trial_hist[1:])
            a.append(trial_hist)
        # np.random.shuffle(a_source)         # shuffling trials
        # np.random.shuffle(a_target)         # shuffling trials
        # np.random.shuffle(a)                # shuffling trials
        spike_train_source = list(itertools.chain.from_iterable(a_source))
        spike_train_target = list(itertools.chain.from_iterable(a_target))
        spike_train        = list(itertools.chain.from_iterable(a))
        if datatype == 'real_data':
            Area_source.append(spike_train_source)            # real data
            Area_target.append(spike_train_target)            # real data
            Area.append(spike_train)                          # real data
        elif datatype == 'residuals':
            Area_source.append(spike_train_source-np.tile(np.mean(a_source, axis=0), len(spike_train_source)//10))    # residuals
            Area_target.append(spike_train_target-np.tile(np.mean(a_target, axis=0), len(spike_train_target)//10))    # residuals
            Area.append(spike_train-np.tile(np.mean(a, axis=0), len(spike_train)//10))                                # residuals

        elif datatype == 'deducted_baseline':
            b = 0
            for time in stims:
                b += len(spike_times[(spike_times>=time+.25) & (spike_times<time+.75)])/20
            Area_source.append(np.array(spike_train_source) - b/len(stims))                 # deducted baseline
            Area_target.append(np.array(spike_train_target) - b/len(stims))                 # deducted baseline
            Area_target.append(np.array(spike_train) - b/len(stims))                        # deducted baseline
    Area_source = np.array(Area_source)
    Area_target = np.array(Area_target)
    Area        = np.array(Area)
    return Area_source, Area_target, Area

def folded_Reduced_Rank(matrix1, matrix2, nfold=10, ptype='regular'):
    if ptype == 'causal':
        nbins = 9
    else:
        nbins = 10
        
    rankvals = np.arange(9)+1          # internal rank/bottleneck
    lamvals  = 2.0**(np.arange(14))    # regularization on the model

    SPLIT  = len(matrix1[0])//nbins//nfold*nbins
    matrix1, matrix2 = matrix1[:, :nfold*SPLIT], matrix2[:, :nfold*SPLIT]
    XX, YY = matrix1.T, matrix2.T
    prediction, prediction_err, prediction_sem = [], [], []
    for RANK in rankvals:
        P, PE, V = [], [], []
        for fold in range(nfold):
            initial_index, final_index = fold*SPLIT, (fold+1)*SPLIT
            testXX,  testYY  = XX[initial_index:final_index], YY[initial_index:final_index]
            trainXX, trainYY = np.delete(XX, np.arange(initial_index, final_index), axis=0), np.delete(YY, np.arange(initial_index, final_index), axis=0)
            training_error, testing_error = [], []
            for LAMBDA in lamvals:
                regressor = reduced_rank_regressor.RRR(trainXX, trainYY, RANK, LAMBDA)
                training_error.append(sqerr(regressor.predict(trainXX), trainYY))
                testing_error.append(sqerr(regressor.predict(testXX), testYY))
            bestLAM   = lamvals[np.argmin(testing_error)]
            regressor = reduced_rank_regressor.RRR(trainXX, trainYY, RANK, bestLAM)
            P.append(regressor.predict(testXX))
            PE.append(sqerr(regressor.predict(testXX), testYY))
            if RANK == 9:
                V.append(regressor.get_V())
        prediction.append(np.mean(P, axis=0))
        prediction_sem.append(PE)
        prediction_err.append(np.mean(PE, axis=0))
    return prediction, prediction_err, prediction_sem, V