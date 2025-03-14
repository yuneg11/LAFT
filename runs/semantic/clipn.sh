python scripts/semantic/clipn.py -m checkpoints/clipn/repeat1.pt -d color_mnist -g guide_number -o _results/color_mnist/guide_number/clipn/repeat1.txt
python scripts/semantic/clipn.py -m checkpoints/clipn/repeat2.pt -d color_mnist -g guide_number -o _results/color_mnist/guide_number/clipn/repeat2.txt
python scripts/semantic/clipn.py -m checkpoints/clipn/repeat3.pt -d color_mnist -g guide_number -o _results/color_mnist/guide_number/clipn/repeat3.txt

python scripts/semantic/clipn.py -m checkpoints/clipn/repeat1.pt -d waterbirds  -g guide_bird   -o _results/waterbirds/guide_bird/clipn/repeat1.txt
python scripts/semantic/clipn.py -m checkpoints/clipn/repeat2.pt -d waterbirds  -g guide_bird   -o _results/waterbirds/guide_bird/clipn/repeat2.txt
python scripts/semantic/clipn.py -m checkpoints/clipn/repeat3.pt -d waterbirds  -g guide_bird   -o _results/waterbirds/guide_bird/clipn/repeat3.txt

python scripts/semantic/clipn.py -m checkpoints/clipn/repeat1.pt -d celeba      -g guide_blond  -o _results/celeba/guide_blond/clipn/repeat1.txt
python scripts/semantic/clipn.py -m checkpoints/clipn/repeat2.pt -d celeba      -g guide_blond  -o _results/celeba/guide_blond/clipn/repeat2.txt
python scripts/semantic/clipn.py -m checkpoints/clipn/repeat3.pt -d celeba      -g guide_blond  -o _results/celeba/guide_blond/clipn/repeat3.txt

python scripts/semantic/clipn.py -m checkpoints/clipn/repeat1.pt -d celeba      -g guide_glass  -o _results/celeba/guide_glass/clipn/repeat1.txt
python scripts/semantic/clipn.py -m checkpoints/clipn/repeat2.pt -d celeba      -g guide_glass  -o _results/celeba/guide_glass/clipn/repeat2.txt
python scripts/semantic/clipn.py -m checkpoints/clipn/repeat3.pt -d celeba      -g guide_glass  -o _results/celeba/guide_glass/clipn/repeat3.txt
