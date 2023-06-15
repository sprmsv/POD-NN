datadir='./data/230601/base'

Ls=(5 10 15 20 30 40)

Ds=(1 2 3)

Ws=(10 20 50 100 200 500)

for D in ${Ds[*]}; do
    for W in ${Ws[*]}; do
        for L in ${Ls[*]}; do
            python run.py \
                --datadir $datadir --n_trn 2048 --n_val 512 \
                -L $L \
                -D $D -W $W \
                --epochs 50e03 --lr 5e-02 \
                --normalize-output
        done
    done
done
