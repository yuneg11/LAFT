python scripts/semantic/winclip.py -m ViT-B-16-quickgelu:dfn2b -d color_mnist -g guide_number -s 5 -o results/color_mnist/guide_number/winclip.txt
python scripts/semantic/winclip.py -m ViT-B-16-quickgelu:dfn2b -d waterbirds  -g guide_bird   -s 5 -o results/waterbirds/guide_bird/winclip.txt
python scripts/semantic/winclip.py -m ViT-B-16-quickgelu:dfn2b -d celeba      -g guide_blond  -s 5 -o results/celeba/guide_blond/winclip.txt
python scripts/semantic/winclip.py -m ViT-B-16-quickgelu:dfn2b -d celeba      -g guide_glass  -s 5 -o results/celeba/guide_glass/winclip.txt
