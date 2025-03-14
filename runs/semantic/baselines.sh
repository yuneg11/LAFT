python scripts/semantic/baselines.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g guide_number -k 30 -s 5 -o results/color_mnist/guide_number/baselines/seed42.txt  --mnist-seed 42
python scripts/semantic/baselines.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g guide_number -k 30 -s 5 -o results/color_mnist/guide_number/baselines/seed43.txt  --mnist-seed 43
python scripts/semantic/baselines.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g guide_number -k 30 -s 5 -o results/color_mnist/guide_number/baselines/seed44.txt  --mnist-seed 44
python scripts/semantic/baselines.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g guide_number -k 30 -s 5 -o results/color_mnist/guide_number/baselines/seed45.txt  --mnist-seed 45
python scripts/semantic/baselines.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g guide_number -k 30 -s 5 -o results/color_mnist/guide_number/baselines/seed46.txt  --mnist-seed 46

python scripts/semantic/baselines.py -m ViT-B-16-quickgelu:dfn2b -d waterbirds  -g guide_bird   -k 30 -s 5 -o results/waterbirds/guide_bird/baselines.txt
python scripts/semantic/baselines.py -m ViT-B-16-quickgelu:dfn2b -d celeba      -g guide_blond  -k 30 -s 5 -o results/celeba/guide_blond/baselines.txt
python scripts/semantic/baselines.py -m ViT-B-16-quickgelu:dfn2b -d celeba      -g guide_glass  -k 30 -s 5 -o results/celeba/guide_glass/baselines.txt
