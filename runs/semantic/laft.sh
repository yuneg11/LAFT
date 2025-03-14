python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g guide_number -o results/color_mnist/guide_number/laft/seed42.txt   --mnist-seed 42
python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g guide_number -o results/color_mnist/guide_number/laft/seed43.txt   --mnist-seed 43
python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g guide_number -o results/color_mnist/guide_number/laft/seed44.txt   --mnist-seed 44
python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g guide_number -o results/color_mnist/guide_number/laft/seed45.txt   --mnist-seed 45
python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g guide_number -o results/color_mnist/guide_number/laft/seed46.txt   --mnist-seed 46

python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d waterbirds  -g guide_bird   -o results/waterbirds/guide_bird/laft.txt
python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d celeba      -g guide_blond  -o results/celeba/guide_blond/laft.txt
python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d celeba      -g guide_glass  -o results/celeba/guide_glass/laft.txt


python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g ignore_color -o results/color_mnist/ignore_color/laft/seed42.txt   --mnist-seed 42
python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g ignore_color -o results/color_mnist/ignore_color/laft/seed43.txt   --mnist-seed 43
python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g ignore_color -o results/color_mnist/ignore_color/laft/seed44.txt   --mnist-seed 44
python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g ignore_color -o results/color_mnist/ignore_color/laft/seed45.txt   --mnist-seed 45
python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g ignore_color -o results/color_mnist/ignore_color/laft/seed46.txt   --mnist-seed 46

python scripts/semantic/laft.py -m ViT-B-16-quickgelu:dfn2b -d waterbirds  -g ignore_back  -o results/waterbirds/ignore_back/laft.txt
