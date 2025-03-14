python scripts/semantic/clip_score.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g guide_number -o results/color_mnist/guide_number/clip_score.txt
python scripts/semantic/clip_score.py -m ViT-B-16-quickgelu:dfn2b -d waterbirds  -g guide_bird   -o results/waterbirds/guide_bird/clip_score.txt
python scripts/semantic/clip_score.py -m ViT-B-16-quickgelu:dfn2b -d celeba      -g guide_blond  -o results/celeba/guide_blond/clip_score.txt
python scripts/semantic/clip_score.py -m ViT-B-16-quickgelu:dfn2b -d celeba      -g guide_glass  -o results/celeba/guide_glass/clip_score.txt
