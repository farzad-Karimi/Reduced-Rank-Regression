import sys
import funcs
import warnings
import numpy as np
from pathlib import Path
from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import VisualBehaviorNeuropixelsProjectCache

warnings.filterwarnings("ignore", category=UserWarning)

def process_image(cache, animal):
    cortex = {0:'VISp', 1:'VISl', 2:'VISal', 3:'VISrl', 4:'VISam', 5:'VISpm', 6:'CA1', 7:'LP', 8:'APN'}
    visp_pre,  visp_post  = funcs.population_extractor_lick(cache, animal, 'VISp')
    visl_pre,  visl_post  = funcs.population_extractor_lick(cache, animal, 'VISl')
    visal_pre, visal_post = funcs.population_extractor_lick(cache, animal, 'VISal')
    visrl_pre, visrl_post = funcs.population_extractor_lick(cache, animal, 'VISrl')
    visam_pre, visam_post = funcs.population_extractor_lick(cache, animal, 'VISam')
    vispm_pre, vispm_post = funcs.population_extractor_lick(cache, animal, 'VISpm')
    ca1_pre,   ca1_post   = funcs.population_extractor_lick(cache, animal, 'CA1')
    lp_pre,    lp_post    = funcs.population_extractor_lick(cache, animal, 'LP')
    apn_pre,   apn_post   = funcs.population_extractor_lick(cache, animal, 'APN')
    data_pre  = [visp_pre,  visl_pre,  visal_pre,  visrl_pre,  visam_pre,  vispm_pre,  ca1_pre,  lp_pre,  apn_pre]
    data_post = [visp_post, visl_post, visal_post, visrl_post, visam_post, vispm_post, ca1_post, lp_post, apn_post]
    for r1ndex, region1 in list(cortex.items())[:-1]:
        for r2ndex, region2 in list(cortex.items())[r1ndex+1:]:
            Area1_pre, Area2_pre = data_pre[r1ndex], data_pre[r2ndex]
            min_N = min(len(Area1_pre), len(Area2_pre))
            if min_N >= 10:
                r_err1, r_sem1, V1 = [], [], []
                r_err2, r_sem2, V2 = [], [], []
                for _ in range(20):
                    np.random.shuffle(Area1_pre)
                    np.random.shuffle(Area2_pre)
                    if len(Area2_pre) > len(Area1_pre):
                        area1 = Area1_pre
                        area2 = Area2_pre[:min_N]
                    else:
                        area1 = Area1_pre[:min_N]
                        area2 = Area2_pre
                    results = funcs.folded_Reduced_Rank(area1, area2, nfold=10, ptype='regular')
                    r_err1.append(results[1])
                    r_sem1.append(results[2])
                    V1.append(results[3])
                    results = funcs.folded_Reduced_Rank(area2, area1, nfold=10, ptype='regular')
                    r_err2.append(results[1])
                    r_sem2.append(results[2])
                    V2.append(results[3])
                save_results(animal, region1, region2, r_err1, r_sem1, V1, dtype='pre_lick', N=min_N)
                save_results(animal, region2, region1, r_err2, r_sem2, V2, dtype='pre_lick', N=min_N)
                
                Area1_post, Area2_post = data_post[r1ndex], data_post[r2ndex]
                r_err1, r_sem1, V1 = [], [], []
                r_err2, r_sem2, V2 = [], [], []
                for _ in range(20):
                    np.random.shuffle(Area1_post)
                    np.random.shuffle(Area2_post)
                    if len(Area2_post) > len(Area1_post):
                        area1 = Area1_post
                        area2 = Area2_post[:min_N]
                    else:
                        area1 = Area1_post[:min_N]
                        area2 = Area2_post
                    results = funcs.folded_Reduced_Rank(area1, area2, nfold=10, ptype='regular')
                    r_err1.append(results[1])
                    r_sem1.append(results[2])
                    V1.append(results[3])
                    results = funcs.folded_Reduced_Rank(area2, area1, nfold=10, ptype='regular')
                    r_err2.append(results[1])
                    r_sem2.append(results[2])
                    V2.append(results[3])
                save_results(animal, region1, region2, r_err1, r_sem1, V1, dtype='post_lick', N=min_N)
                save_results(animal, region2, region1, r_err2, r_sem2, V2, dtype='post_lick', N=min_N)

def save_results(animal, r1, r2, r_err, r_sem, V, dtype, N):
    print(r1, r2, animal, N)
    base_path = Path(r"results/inter/{}".format(dtype))
    np.save(base_path / f"{r1}_{r2}/prediction_error/{animal}", np.mean(r_err, axis=0))
    np.save(base_path / f"{r1}_{r2}/prediction_sem/{animal}", np.mean(r_sem, axis=0))
    np.save(base_path / f"{r1}_{r2}/predictive_dimensions/{animal}", np.mean(V, axis=0))

def final(animal):
    output_dir = Path(r"Allen/Data")
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=output_dir)
    process_image(cache, animal)

if __name__ == '__main__':
    final(int(sys.argv[2]))