run -J test -A gl-rg -p tier3 -t 20:0:0 --nodes=1 --ntasks-per-node=4 --cpus-per-task=5 --gres=gpu:a100:1 --mem=100g  --pty bash