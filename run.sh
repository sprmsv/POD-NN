datadir='./data/230601/base'
Ls=(1 5 10 15 20 30 40)
Ds=(1)
Ws=(10 20 50 100 200 500)

for D in ${Ds[*]}; do
    for W in ${Ws[*]}; do
        for L in ${Ls[*]}; do
            python run.py \
                --datadir $datadir --n_trn 2048 --n_val 512 \
                -L $L \
                -D $D -W $W \
                --epochs 500e03 --lr 5e-04 \
                --normalize-output
        done
    done
done
