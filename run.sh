# datadirs=('./data/230601/base' './data/230601/highfreq' './data/230601/moredif' './data/230601/non-smooth')
datadirs=('./data/230601/base')

Ds=(1 2 3)
Ws=(10 20 50 100 200 500)
Ls=(5 10 15 20 30 40)

for datadir in ${datadirs[*]}; do
    for D in ${Ds[*]}; do
        for W in ${Ws[*]}; do
            for L in ${Ls[*]}; do
                python run.py \
                    --datadir $datadir --n_trn 2048 --n_val 512 \
                    -L $L -D $D -W $W \
                    --epochs 200e03 --lr 5e-02 \
                    --normalize-output
            done
        done
    done
done
