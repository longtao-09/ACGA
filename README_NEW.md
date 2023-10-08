#### environment install:

```
pip install -r requirements.txt
```

#### Parameter introduction:

```
--dataset  cora or citeseer or airpotr ...
--hidden_size encoder hidden dimension, node clsssification defalut 128,link prediction defalut 32
--emb_size VGAE/GAE embedding dimension,node clsssification defalut 32,link prediction defalut 16
--gae selct VGAE or GAE
--use_bns use bn
--task 0 is node clsssification, and 1 is link prediction
--alpha
--beta
--gamma optuna get best parameter
```



#### Run parameter search node classification：

```
python optuna_ACGA.py --dataset cora --hidden_size 128 --emb_size 32 --gae 0 --use_bns True --task 0
```

#### Run parameter search link prediction：

```
python optuna_ACGA.py --dataset cora --hidden_size 32 --emb_size 16 --gae 0 --use_bns True --task 1
```



#### Run the node classification task：

```
python ACGA.py --dataset cora --hidden_size 128 --emb_size 32 --gae 0 --use_bns True --task 0
```

#### Run the link prediction task：

```
python ACGA.py --dataset cora --hidden_size 32 --emb_size 16 --gae 0 --use_bns True --task 1
```

