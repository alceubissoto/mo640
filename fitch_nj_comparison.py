import pandas as pd
from os import listdir
'''
Instructions:
1) run this script
2) upload them to https://docs.google.com/spreadsheets/d/1aYoXsroiGxo2dtmMH8_s5NrIunWNN9Ugu413c9aCylE/edit#gid=730202918
2.1) ends.csv should be copy/pasted to tab "algorithm last fitness" 
2.2) fitch_nj.csv should be copy/pasted to tab "nj and fitch pure"
'''

df = pd.DataFrame(columns=['iteration', 'timestamp', 'fitness', 'dataset', 'matrix', 'algo'])

# append all
for i, file in enumerate(listdir('final_results/')):
    if file.endswith('_100.csv'):
        new_df = pd.read_csv('final_results/' + file,
                             names=['iteration', 'timestamp', 'fitness', 'dataset', 'matrix', 'algo'])
        df = df.append(new_df)

ends = df.groupby(['algo', 'matrix']).min()['fitness']
ends.to_csv('ends.csv')

df = pd.read_csv('results_phylip.csv',
                 names=['matrix', 'location', 'nj_fitness', 'fitch_fitness'],
                 dtype={'matrix': object}
                 )
df['matrix'] = df['matrix'].astype(str) + 'additive.noisy.npy'
df.set_index('matrix', inplace=True)
df.to_csv('fitch_nj.csv')

