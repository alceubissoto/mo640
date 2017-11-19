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

## Strategy 1
- input: non-additive matrix
- create mutated tree
- add weights to the tree
- calculate fitness
- go back to step 2

## Strategy 2
- non-additive matrix

 

## Seed matrix used
```
    a     b     c     d     e
a   0    12    12    12    12
b   12    0     4     6     6
c   12    4     0     6     6
d   12    6     6     0     2
e   12    6     6     2     0

matrix[1][0] = 12

matrix = np.array([[0, 0, 12, 12, 12, 12],
                   [1, 12, 0, 4, 6, 6],
                   [2, 12, 4, 0, 6, 6],
                   [3, 12, 6, 6, 0, 2],
                   [4, 12, 6, 6, 2, 0]])
```