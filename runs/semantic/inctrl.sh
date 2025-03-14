python scripts/semantic/inctrl.py -m checkpoints/inctrl/mvtec2.pt -d color_mnist -g guide_number -o results/color_mnist/guide_number/inctrl/mvtec2.txt
python scripts/semantic/inctrl.py -m checkpoints/inctrl/mvtec4.pt -d color_mnist -g guide_number -o results/color_mnist/guide_number/inctrl/mvtec4.txt
python scripts/semantic/inctrl.py -m checkpoints/inctrl/mvtec8.pt -d color_mnist -g guide_number -o results/color_mnist/guide_number/inctrl/mvtec8.txt

python scripts/semantic/inctrl.py -m checkpoints/inctrl/mvtec2.pt -d waterbirds  -g guide_bird   -o results/waterbirds/guide_bird/inctrl/mvtec2.txt
python scripts/semantic/inctrl.py -m checkpoints/inctrl/mvtec4.pt -d waterbirds  -g guide_bird   -o results/waterbirds/guide_bird/inctrl/mvtec4.txt
python scripts/semantic/inctrl.py -m checkpoints/inctrl/mvtec8.pt -d waterbirds  -g guide_bird   -o results/waterbirds/guide_bird/inctrl/mvtec8.txt

python scripts/semantic/inctrl.py -m checkpoints/inctrl/mvtec2.pt -d celeba      -g guide_blond  -o results/celeba/guide_blond/inctrl/mvtec2.txt
python scripts/semantic/inctrl.py -m checkpoints/inctrl/mvtec4.pt -d celeba      -g guide_blond  -o results/celeba/guide_blond/inctrl/mvtec4.txt
python scripts/semantic/inctrl.py -m checkpoints/inctrl/mvtec8.pt -d celeba      -g guide_blond  -o results/celeba/guide_blond/inctrl/mvtec8.txt

python scripts/semantic/inctrl.py -m checkpoints/inctrl/mvtec2.pt -d celeba      -g guide_glass  -o results/celeba/guide_glass/inctrl/mvtec2.txt
python scripts/semantic/inctrl.py -m checkpoints/inctrl/mvtec4.pt -d celeba      -g guide_glass  -o results/celeba/guide_glass/inctrl/mvtec4.txt
python scripts/semantic/inctrl.py -m checkpoints/inctrl/mvtec8.pt -d celeba      -g guide_glass  -o results/celeba/guide_glass/inctrl/mvtec8.txt
