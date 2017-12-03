# to generate the preliminary datasets
python3.6 cgp.py --out-data=data_n10_std10 --num-leafs=10 --stdev=10 --max-weight=100 --qty=10
python3.6 cgp.py --out-data=data_n10_std25 --num-leafs=10 --stdev=25 --max-weight=100 --qty=10
python3.6 cgp.py --out-data=data_n10_std50 --num-leafs=10 --stdev=50 --max-weight=100 --qty=10
python3.6 cgp.py --out-data=data_n20_std10 --num-leafs=20 --stdev=10 --max-weight=100 --qty=10
python3.6 cgp.py --out-data=data_n20_std25 --num-leafs=20 --stdev=25 --max-weight=100 --qty=10
python3.6 cgp.py --out-data=data_n20_std50 --num-leafs=20 --stdev=50 --max-weight=100 --qty=10
python3.6 cgp.py --out-data=data_n30_std10 --num-leafs=30 --stdev=10 --max-weight=100 --qty=10
python3.6 cgp.py --out-data=data_n30_std25 --num-leafs=30 --stdev=25 --max-weight=100 --qty=10
python3.6 cgp.py --out-data=data_n30_std50 --num-leafs=30 --stdev=50 --max-weight=100 --qty=10

# to generate the final datasets
python3.6 cgp.py --out-data=data_n05_100 --num-leafs=5 --stdev=25 --max-weight=100 --qty=100
python3.6 cgp.py --out-data=data_n10_100 --num-leafs=10 --stdev=25 --max-weight=100 --qty=100
python3.6 cgp.py --out-data=data_n15_100 --num-leafs=15 --stdev=25 --max-weight=100 --qty=100
python3.6 cgp.py --out-data=data_n20_100 --num-leafs=20 --stdev=25 --max-weight=100 --qty=100