echo "Creating dataset..."
rm -rf data
python3.6 cgp.py --out-data=data
echo "DONE!"

echo "Running experiments..."
python3.6 cgp.py --in-data=data
echo "DONE!"
