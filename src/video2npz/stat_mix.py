def _cal_density(flow_magnitude):
    for i, percentile in enumerate(fmpb_percentile):
        if flow_magnitude < percentile:
            return i
    return len(fmpb_percentile)


def _cal_strength(weight):
    for i, percentile in enumerate(vbeat_weight_percentile):
        if weight < percentile:
            return i
    return len(vbeat_weight_percentile)


# calculated based on a set of videos

vbeat_weight_percentile = [0, 0.22890276357193542, 0.4838207191278801, 0.7870981363596372, 0.891160136856027,
                           0.9645568135300789, 0.991241869205911, 0.9978208223154553, 0.9996656159745393,
                           0.9998905521344276]
fmpb_percentile = [0.008169269189238548, 0.020344337448477745, 0.02979462407529354, 0.041041795164346695,
                   0.07087484002113342, 0.10512548685073853, 0.14267262816429138, 0.19095642864704132,
                   0.5155120491981506, 0.7514784336090088, 0.9989343285560608, 1.2067525386810303, 1.6322582960128784,
                   2.031705141067505, 2.467430591583252, 2.8104422092437744]