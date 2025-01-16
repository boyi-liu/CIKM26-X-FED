md=transformer
dts=agnews
sr=0.1
rnd=100
cnum=100
ep=5
lr=0.1

algorithms=("local" "fedavg" "fedprox" "fedper" "fedrep" "fedbabu" "heurfedamp" "pfedgraph" "fedrod" "ditto" "fedpac" "xfed")
 modes=("jt")
suffix="agnews-official"

for (( i=0; i<${#algorithms[@]}; i++ )); do
    alg=${algorithms[$i]}
    for mode in "${modes[@]}"; do
        python ../main.py --alg $alg --mode $mode --device 0 --alpha 0.15 --lam 0.2 --suffix "$suffix" --dataset $dts --model $md --sr $sr --rnd $rnd --total_num $cnum --epoch $ep --ee_num 4 --lr $lr
    done
done