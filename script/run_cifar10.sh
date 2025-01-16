md=convnet
dts=cifar10
sr=0.1
rnd=300
cnum=100
ep=5
lr=0.1

algorithms=("local" "fedavg" "fedprox" "fedper" "fedrep" "fedbabu" "heurfedamp" "pfedgraph" "fedrod" "ditto" "fedpac" "xfed")
mode="jt"

suffix="cifar10-official"

for (( i=0; i<${#algorithms[@]}; i++ )); do
    alg=${algorithms[$i]}
    python ../main.py --alg $alg --mode $mode --device 0 --suffix "$suffix" --dataset $dts --model $md --sr $sr --rnd $rnd --total_num $cnum --epoch $ep --lr $lr --ee_num 3 &
done