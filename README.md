# mo640

## How to use
 Create distance matrix: 
```
$ python cgp.py dataset
```

Run algorithm with input:
```
$ python cgp.py <path_input>'
```

## Files generated
- *.additive.npy - additive distance matrix in npy format
- *.additive.txt - same matrix in human readable format
- *.noisy.npy - the additive matrix above but with gaussian noise 
applied
- *.noisy.txt - same as noisy.npy but in human readable format
- *.jpg - graph chart
- *.spring.jpg - same as above but in a layout that might be better
