import json
import math
import pdb

from tqdm import tqdm


def _cal_b_density(flow_magnitude):
    for i, percentile in enumerate(fmpb_percentile):
        if flow_magnitude < percentile:
            return i
    return len(fmpb_percentile)

def _cal_o_density(weight):
    for i, percentile in enumerate(vbeat_weight_percentile):
        if weight < percentile:
            return i
    return len(vbeat_weight_percentile)

vbeat_weight_percentile = [0, 0.22890276357193542, 0.4838207191278801, 0.7870981363596372, 0.891160136856027, 0.9645568135300789, 0.991241869205911, 0.9978208223154553, 0.9996656159745393, 0.9998905521344276]
fmpb_percentile = [0.008169269189238548, 0.020344337448477745, 0.02979462407529354, 0.041041795164346695, 0.07087484002113342, 0.10512548685073853, 0.14267262816429138, 0.19095642864704132, 0.5155120491981506, 0.7514784336090088, 0.9989343285560608, 1.2067525386810303, 1.6322582960128784, 2.031705141067505, 2.467430591583252, 2.8104422092437744]

################# o_dens #################
# o_dens_percentage = {
#     'Piano': [

#     ]
# }

# [
#     0.5438, # 1
#     0.2305, # 2
#     0.1337, # 3
#     0.0577, # 4
#     0.0213, # 5
#     0.0090, # 6
#     0.0025, # 7
#     0.0009, # 8
#     0.0003  # 9
# ]
# '''
# o_dens: 1, 7868576 / 14468678, 54.38%
# o_dens: 2, 3334852 / 14468678, 23.05%
# o_dens: 3, 1934401 / 14468678, 13.37%
# o_dens: 4, 835012 / 14468678, 5.77%
# o_dens: 5, 308613 / 14468678, 2.13%
# o_dens: 6, 130653 / 14468678, 0.90%
# o_dens: 7, 36176 / 14468678, 0.25%
# o_dens: 8, 13118 / 14468678, 0.09%
# o_dens: 9, 4381 / 14468678, 0.03%
# o_dens: 10, 1694 / 14468678, 0.01%
# o_dens: 11, 549 / 14468678, 0.00%
# o_dens: 12, 383 / 14468678, 0.00%
# o_dens: 13, 92 / 14468678, 0.00%
# o_dens: 14, 33 / 14468678, 0.00%
# o_dens: 15, 59 / 14468678, 0.00%
# o_dens: 16, 21 / 14468678, 0.00%
# o_dens: 17, 1 / 14468678, 0.00%
# o_dens: 18, 61 / 14468678, 0.00%
# o_dens: 19, 3 / 14468678, 0.00%
# '''

# vbeat_weight_list = []
# with open('metadata_v2.json', 'r') as f:
#     metadata = json.load(f)
# for _, video_metadata in tqdm(metadata.items()):
#     for vbeat in video_metadata['vbeats']:
#         vbeat_weight_list.append(vbeat['weight'])

# vbeat_weight_list = sorted(vbeat_weight_list)
# num = len(vbeat_weight_list)
# vbeat_weight_percentile = [0]
# sum_percentage = 0.
# for percentage in o_dens_percentage:
#     sum_percentage += percentage
#     vbeat_weight_percentile.append(vbeat_weight_list[math.floor(sum_percentage * num)])
# for i, percentile in enumerate(vbeat_weight_percentile):
#     print("%d\t%f" % (i, percentile))

# print(vbeat_weight_percentile)
# '''
# 0       0.000000
# 1       0.228903
# 2       0.483821
# 3       0.787098
# 4       0.891160
# 5       0.964557
# 6       0.991242
# 7       0.997821
# 8       0.999666
# 9       0.999891
# '''


# ################# b_dens #################
# b_dens_percentage = [
#     0.0195, # 1
#     0.0091, # 2
#     0.0065, # 3
#     0.0075, # 4
#     0.0193, # 5
#     0.0234, # 6
#     0.0334, # 7
#     0.0443, # 8
#     0.2709, # 9
#     0.1173, # 10
#     0.0788, # 11
#     0.0525, # 12
#     0.0704, # 13
#     0.0446, # 14
#     0.0356, # 15
#     0.0278  # 16
# ]
# '''
# b_dens: 1, 12998 / 665612, 1.95%
# b_dens: 2, 6042 / 665612, 0.91%
# b_dens: 3, 4340 / 665612, 0.65%
# b_dens: 4, 4978 / 665612, 0.75%
# b_dens: 5, 12847 / 665612, 1.93%
# b_dens: 6, 15565 / 665612, 2.34%
# b_dens: 7, 22219 / 665612, 3.34%
# b_dens: 8, 29463 / 665612, 4.43%
# b_dens: 9, 180284 / 665612, 27.09%
# b_dens: 10, 78078 / 665612, 11.73%
# b_dens: 11, 52481 / 665612, 7.88%
# b_dens: 12, 34962 / 665612, 5.25%
# b_dens: 13, 46884 / 665612, 7.04%
# b_dens: 14, 29673 / 665612, 4.46%
# b_dens: 15, 23716 / 665612, 3.56%
# b_dens: 16, 18494 / 665612, 2.78%
# b_dens: 17, 92588 / 665612, 13.91%
# '''

# fmpb_list = []
# with open('metadata_v2.json', 'r') as f:
#     metadata = json.load(f)
# for _, video_metadata in tqdm(metadata.items()):
#     fmpb_list += video_metadata['flow_magnitude_per_bar']

# fmpb_list = sorted(fmpb_list)
# num = len(fmpb_list)
# fmpb_percentile = []
# sum_percentage = 0.
# for percentage in b_dens_percentage:
#     sum_percentage += percentage
#     fmpb_percentile.append(fmpb_list[math.floor(sum_percentage * num)])
# for i, percentile in enumerate(fmpb_percentile):
#     print("%d\t%f" % (i, percentile))

# print(fmpb_percentile)
# '''
# 0       0.008169
# 1       0.020344
# 2       0.029795
# 3       0.041042
# 4       0.070875
# 5       0.105125
# 6       0.142673
# 7       0.190956
# 8       0.515512
# 9       0.751478
# 10      0.998934
# 11      1.206753
# 12      1.632258
# 13      2.031705
# 14      2.467431
# 15      2.810442
# '''


# ################# visbeat per bar #################
# import matplotlib.pyplot as plt
# import math

# n_visbeat_per_bar = [0] * 110
# n_visbeats = 0
# n_bars = 0
# with open('metadata_v2.json', 'r') as f:
#     metadata = json.load(f)
# for _, video_metadata in tqdm(metadata.items()):
#     n_visbeats += len(video_metadata['vbeats'])
#     n_bars += math.ceil(video_metadata['duration'] / 60 * video_metadata['tempo'] / 4)
#     for visbeat in video_metadata['vbeats']:
#         n_visbeat_per_bar[int(visbeat['bar'])] += 1
# plt.figure(figsize=(20, 10))
# plt.hist(n_visbeat_per_bar, bins=110, facecolor='blue', edgecolor='black', alpha=0.7)
# plt.savefig('lalal.jpg')
# print('avg. visbeats per bar: %f' % (n_visbeats / n_bars))
