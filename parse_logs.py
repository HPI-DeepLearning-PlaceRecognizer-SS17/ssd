import sys
import os
import pandas as pd


dic = {}
with open(sys.argv[1], 'r') as inpfile:
    for line in inpfile:
        if 'Validation' in line:
            split = line.split('-')
            split = split[1].split('=')
            tag = split[0]
            if tag != 'mAP':
                tag = 'mAP_%s' % tag
            else:
                tag = 'total_mAP'
            if not tag in dic:
                dic[tag] = []
            dic[tag].append(float(split[1]))

pd.DataFrame(dic).to_csv(sys.argv[2])
