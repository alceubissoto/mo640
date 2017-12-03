rm ends.csv
rm fitch_nj.csv
rm results_phylip.csv
python3.6 phylip.py --input=data_n05_100
python3.6 phylip.py --input=data_n10_100
python3.6 phylip.py --input=data_n15_100
python3.6 phylip.py --input=data_n20_100
python3.6 fitch_nj_comparison.py