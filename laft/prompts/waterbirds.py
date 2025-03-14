DATASET_CONFIG = {
    "bird": {
        "land": True,    # Anomaly
        "water": False,  # Normal
    },
    "background": {
        "land": True,    # Anomaly
        "water": False,  # Normal
    }
}

BIRD_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
    "a photo of a {}, a type of bird.",
    "a blurry photo of a {}, a type of bird.",
    "a low contrast photo of a {}, a type of bird.",
    "a high contrast photo of a {}, a type of bird.",
    "a bad photo of a {}, a type of bird.",
    "a good photo of a {}, a type of bird.",
    "a photo of a small {}, a type of bird.",
    "a photo of a big {}, a type of bird.",
    "a photo of the {}, a type of bird.",
    "a blurry photo of the {}, a type of bird.",
    "a bad photo of the {}, a type of bird.",
    "a good photo of the {}, a type of bird.",
    "a photo of the small {}, a type of bird.",
    "a photo of the big {}, a type of bird.",
]
BACK_TEMPLATES = [
    v + u
    for v in
    [
        "a photo of a {}",
        "a photo of a bird with {}",
        "a blurry photo of a {}",
        "a blurry photo of a bird with {}",
        "a bad photo of a {}",
        "a bad photo of a bird with {}",
        "a good photo of a {}",
        "a good photo of a bird with {}",
        "a photo of the {}",
        "a photo of the bird with {}",
        "a bad photo of the {}",
        "a bad photo of the bird with {}",
        "a good photo of the {}",
        "a good photo of the bird with {}",
    ]
    for u in [
        "",
        " background.",
        " scene.",
        " place.",
        " setting.",
        " environment.",
        " surroundings.",
    ]
]

LAND_BIRD_WORDS = [
    "Groove billed Ani",
    "Brewer Blackbird",
    "Red winged Blackbird",
    "Rusty Blackbird",
    "Yellow headed Blackbird",
    "Bobolink",
    "Indigo Bunting",
    "Lazuli Bunting",
    "Painted Bunting",
    "Cardinal",
    "Spotted Catbird",
    "Gray Catbird",
    "Yellow breasted Chat",
    "Chuck will Widow",
    "Bronzed Cowbird",
    "Shiny Cowbird",
    "Brown Creeper",
    "American Crow",
    "Fish Crow",
    "Black billed Cuckoo",
    "Mangrove Cuckoo",
    "Yellow billed Cuckoo",
    "Gray crowned Rosy Finch",
    "Purple Finch",
    "Northern Flicker",
    "Acadian Flycatcher",
    "Great Crested Flycatcher",
    "Least Flycatcher",
    "Olive sided Flycatcher",
    "Scissor tailed Flycatcher",
    "Vermilion Flycatcher",
    "Yellow bellied Flycatcher",
    "American Goldfinch",
    "European Goldfinch",
    "Boat tailed Grackle",
    "Blue Grosbeak",
    "Evening Grosbeak",
    "Pine Grosbeak",
    "Rose breasted Grosbeak",
    "Anna Hummingbird",
    "Ruby throated Hummingbird",
    "Rufous Hummingbird",
    "Green Violetear",
    "Blue Jay",
    "Florida Jay",
    "Green Jay",
    "Dark eyed Junco",
    "Tropical Kingbird",
    "Gray Kingbird",
    "Belted Kingfisher",
    "Green Kingfisher",
    "Pied Kingfisher",
    "Ringed Kingfisher",
    "White breasted Kingfisher",
    "Horned Lark",
    "Mockingbird",
    "Nighthawk",
    "Clark Nutcracker",
    "White breasted Nuthatch",
    "Baltimore Oriole",
    "Hooded Oriole",
    "Orchard Oriole",
    "Scott Oriole",
    "Ovenbird",
    "Sayornis",
    "American Pipit",
    "Whip poor Will",
    "Common Raven",
    "White necked Raven",
    "American Redstart",
    "Geococcyx",
    "Loggerhead Shrike",
    "Great Grey Shrike",
    "Baird Sparrow",
    "Black throated Sparrow",
    "Brewer Sparrow",
    "Chipping Sparrow",
    "Clay colored Sparrow",
    "House Sparrow",
    "Field Sparrow",
    "Fox Sparrow",
    "Grasshopper Sparrow",
    "Harris Sparrow",
    "Henslow Sparrow",
    "Le Conte Sparrow",
    "Lincoln Sparrow",
    "Nelson Sharp tailed Sparrow",
    "Savannah Sparrow",
    "Seaside Sparrow",
    "Song Sparrow",
    "Tree Sparrow",
    "Vesper Sparrow",
    "White crowned Sparrow",
    "White throated Sparrow",
    "Cape Glossy Starling",
    "Bank Swallow",
    "Barn Swallow",
    "Cliff Swallow",
    "Tree Swallow",
    "Scarlet Tanager",
    "Summer Tanager",
    "Green tailed Towhee",
    "Brown Thrasher",
    "Sage Thrasher",
    "Black capped Vireo",
    "Blue headed Vireo",
    "Philadelphia Vireo",
    "Red eyed Vireo",
    "Warbling Vireo",
    "White eyed Vireo",
    "Yellow throated Vireo",
    "Bay breasted Warbler",
    "Black and white Warbler",
    "Black throated Blue Warbler",
    "Blue winged Warbler",
    "Canada Warbler",
    "Cape May Warbler",
    "Cerulean Warbler",
    "Chestnut sided Warbler",
    "Golden winged Warbler",
    "Hooded Warbler",
    "Kentucky Warbler",
    "Magnolia Warbler",
    "Mourning Warbler",
    "Myrtle Warbler",
    "Nashville Warbler",
    "Orange crowned Warbler",
    "Palm Warbler",
    "Pine Warbler",
    "Prairie Warbler",
    "Prothonotary Warbler",
    "Swainson Warbler",
    "Tennessee Warbler",
    "Wilson Warbler",
    "Worm eating Warbler",
    "Yellow Warbler",
    "Northern Waterthrush",
    "Louisiana Waterthrush",
    "Bohemian Waxwing",
    "Cedar Waxwing",
    "American Three toed Woodpecker",
    "Pileated Woodpecker",
    "Red bellied Woodpecker",
    "Red cockaded Woodpecker",
    "Red headed Woodpecker",
    "Downy Woodpecker",
    "Bewick Wren",
    "Cactus Wren",
    "Carolina Wren",
    "House Wren",
    "Marsh Wren",
    "Rock Wren",
    "Winter Wren",
    "Common Yellowthroat",
]
WATER_BIRD_WORDS = [
    "Black footed Albatross",
    "Laysan Albatross",
    "Sooty Albatross",
    "Crested Auklet",
    "Least Auklet",
    "Parakeet Auklet",
    "Rhinoceros Auklet",
    "Eastern Towhee",
    "Brandt Cormorant",
    "Red faced Cormorant",
    "Pelagic Cormorant",
    "Frigatebird",
    "Northern Fulmar",
    "Gadwall",
    "Eared Grebe",
    "Horned Grebe",
    "Pied billed Grebe",
    "Western Grebe",
    "Pigeon Guillemot",
    "California Gull",
    "Glaucous winged Gull",
    "Heermann Gull",
    "Herring Gull",
    "Ivory Gull",
    "Ring billed Gull",
    "Slaty backed Gull",
    "Western Gull",
    "Long tailed Jaeger",
    "Pomarine Jaeger",
    "Red legged Kittiwake",
    "Pacific Loon",
    "Mallard",
    "Western Meadowlark",
    "Hooded Merganser",
    "Red breasted Merganser",
    "Brown Pelican",
    "White Pelican",
    "Western Wood Pewee",
    "Horned Puffin",
    "Artic Tern",
    "Black Tern",
    "Caspian Tern",
    "Common Tern",
    "Elegant Tern",
    "Forsters Tern",
    "Least Tern",
]
# Birdsnap (OpenAI/CLIP)
AUX_BIRD_WORDS = [
    "Acadian Flycatcher",
    "Acorn Woodpecker",
    "Alder Flycatcher",
    "Allens Hummingbird",
    "Altamira Oriole",
    "American Avocet",
    "American Bittern",
    "American Black Duck",
    "American Coot",
    "American Crow",
    "American Dipper",
    "American Golden Plover",
    "American Goldfinch",
    "American Kestrel",
    "American Oystercatcher",
    "American Pipit",
    "American Redstart",
    "American Robin",
    "American Three toed Woodpecker",
    "American Tree Sparrow",
    "American White Pelican",
    "American Wigeon",
    "American Woodcock",
    "Anhinga",
    "Annas Hummingbird",
    "Arctic Tern",
    "Ash throated Flycatcher",
    "Audubons Oriole",
    "Bairds Sandpiper",
    "Bald Eagle",
    "Baltimore Oriole",
    "Band tailed Pigeon",
    "Barn Swallow",
    "Barred Owl",
    "Barrows Goldeneye",
    "Bay breasted Warbler",
    "Bells Vireo",
    "Belted Kingfisher",
    "Bewicks Wren",
    "Black Guillemot",
    "Black Oystercatcher",
    "Black Phoebe",
    "Black Rosy Finch",
    "Black Scoter",
    "Black Skimmer",
    "Black Tern",
    "Black Turnstone",
    "Black Vulture",
    "Black and white Warbler",
    "Black backed Woodpecker",
    "Black bellied Plover",
    "Black billed Cuckoo",
    "Black billed Magpie",
    "Black capped Chickadee",
    "Black chinned Hummingbird",
    "Black chinned Sparrow",
    "Black crested Titmouse",
    "Black crowned Night Heron",
    "Black headed Grosbeak",
    "Black legged Kittiwake",
    "Black necked Stilt",
    "Black throated Blue Warbler",
    "Black throated Gray Warbler",
    "Black throated Green Warbler",
    "Black throated Sparrow",
    "Blackburnian Warbler",
    "Blackpoll Warbler",
    "Blue Grosbeak",
    "Blue Jay",
    "Blue gray Gnatcatcher",
    "Blue headed Vireo",
    "Blue winged Teal",
    "Blue winged Warbler",
    "Boat tailed Grackle",
    "Bobolink",
    "Bohemian Waxwing",
    "Bonapartes Gull",
    "Boreal Chickadee",
    "Brandts Cormorant",
    "Brant",
    "Brewers Blackbird",
    "Brewers Sparrow",
    "Bridled Titmouse",
    "Broad billed Hummingbird",
    "Broad tailed Hummingbird",
    "Broad winged Hawk",
    "Bronzed Cowbird",
    "Brown Creeper",
    "Brown Pelican",
    "Brown Thrasher",
    "Brown capped Rosy Finch",
    "Brown crested Flycatcher",
    "Brown headed Cowbird",
    "Brown headed Nuthatch",
    "Bufflehead",
    "Bullocks Oriole",
    "Burrowing Owl",
    "Bushtit",
    "Cackling Goose",
    "Cactus Wren",
    "California Gull",
    "California Quail",
    "California Thrasher",
    "California Towhee",
    "Calliope Hummingbird",
    "Canada Goose",
    "Canada Warbler",
    "Canvasback",
    "Canyon Towhee",
    "Canyon Wren",
    "Cape May Warbler",
    "Carolina Chickadee",
    "Carolina Wren",
    "Caspian Tern",
    "Cassins Finch",
    "Cassins Kingbird",
    "Cassins Sparrow",
    "Cassins Vireo",
    "Cattle Egret",
    "Cave Swallow",
    "Cedar Waxwing",
    "Cerulean Warbler",
    "Chestnut backed Chickadee",
    "Chestnut collared Longspur",
    "Chestnut sided Warbler",
    "Chihuahuan Raven",
    "Chimney Swift",
    "Chipping Sparrow",
    "Cinnamon Teal",
    "Clapper Rail",
    "Clarks Grebe",
    "Clarks Nutcracker",
    "Clay colored Sparrow",
    "Cliff Swallow",
    "Common Black Hawk",
    "Common Eider",
    "Common Gallinule",
    "Common Goldeneye",
    "Common Grackle",
    "Common Ground Dove",
    "Common Loon",
    "Common Merganser",
    "Common Murre",
    "Common Nighthawk",
    "Common Raven",
    "Common Redpoll",
    "Common Tern",
    "Common Yellowthroat",
    "Connecticut Warbler",
    "Coopers Hawk",
    "Cordilleran Flycatcher",
    "Costas Hummingbird",
    "Couchs Kingbird",
    "Crested Caracara",
    "Curve billed Thrasher",
    "Dark eyed Junco",
    "Dickcissel",
    "Double crested Cormorant",
    "Downy Woodpecker",
    "Dunlin",
    "Dusky Flycatcher",
    "Dusky Grouse",
    "Eared Grebe",
    "Eastern Bluebird",
    "Eastern Kingbird",
    "Eastern Meadowlark",
    "Eastern Phoebe",
    "Eastern Screech Owl",
    "Eastern Towhee",
    "Eastern Wood Pewee",
    "Elegant Trogon",
    "Elf Owl",
    "Eurasian Collared Dove",
    "Eurasian Wigeon",
    "European Starling",
    "Evening Grosbeak",
    "Ferruginous Hawk",
    "Ferruginous Pygmy Owl",
    "Field Sparrow",
    "Fish Crow",
    "Florida Scrub Jay",
    "Forsters Tern",
    "Fox Sparrow",
    "Franklins Gull",
    "Fulvous Whistling Duck",
    "Gadwall",
    "Gambels Quail",
    "Gila Woodpecker",
    "Glaucous Gull",
    "Glaucous winged Gull",
    "Glossy Ibis",
    "Golden Eagle",
    "Golden crowned Kinglet",
    "Golden crowned Sparrow",
    "Golden fronted Woodpecker",
    "Golden winged Warbler",
    "Grasshopper Sparrow",
    "Gray Catbird",
    "Gray Flycatcher",
    "Gray Jay",
    "Gray Kingbird",
    "Gray cheeked Thrush",
    "Gray crowned Rosy Finch",
    "Great Black backed Gull",
    "Great Blue Heron",
    "Great Cormorant",
    "Great Crested Flycatcher",
    "Great Egret",
    "Great Gray Owl",
    "Great Horned Owl",
    "Great Kiskadee",
    "Great tailed Grackle",
    "Greater Prairie Chicken",
    "Greater Roadrunner",
    "Greater Sage Grouse",
    "Greater Scaup",
    "Greater White fronted Goose",
    "Greater Yellowlegs",
    "Green Jay",
    "Green tailed Towhee",
    "Green winged Teal",
    "Groove billed Ani",
    "Gull billed Tern",
    "Hairy Woodpecker",
    "Hammonds Flycatcher",
    "Harlequin Duck",
    "Harriss Hawk",
    "Harriss Sparrow",
    "Heermanns Gull",
    "Henslows Sparrow",
    "Hepatic Tanager",
    "Hermit Thrush",
    "Herring Gull",
    "Hoary Redpoll",
    "Hooded Merganser",
    "Hooded Oriole",
    "Hooded Warbler",
    "Horned Grebe",
    "Horned Lark",
    "House Finch",
    "House Sparrow",
    "House Wren",
    "Huttons Vireo",
    "Iceland Gull",
    "Inca Dove",
    "Indigo Bunting",
    "Killdeer",
    "King Rail",
    "Ladder backed Woodpecker",
    "Lapland Longspur",
    "Lark Bunting",
    "Lark Sparrow",
    "Laughing Gull",
    "Lazuli Bunting",
    "Le Contes Sparrow",
    "Least Bittern",
    "Least Flycatcher",
    "Least Grebe",
    "Least Sandpiper",
    "Least Tern",
    "Lesser Goldfinch",
    "Lesser Nighthawk",
    "Lesser Scaup",
    "Lesser Yellowlegs",
    "Lewiss Woodpecker",
    "Limpkin",
    "Lincolns Sparrow",
    "Little Blue Heron",
    "Loggerhead Shrike",
    "Long billed Curlew",
    "Long billed Dowitcher",
    "Long billed Thrasher",
    "Long eared Owl",
    "Long tailed Duck",
    "Louisiana Waterthrush",
    "Magnificent Frigatebird",
    "Magnolia Warbler",
    "Mallard",
    "Marbled Godwit",
    "Marsh Wren",
    "Merlin",
    "Mew Gull",
    "Mexican Jay",
    "Mississippi Kite",
    "Monk Parakeet",
    "Mottled Duck",
    "Mountain Bluebird",
    "Mountain Chickadee",
    "Mountain Plover",
    "Mourning Dove",
    "Mourning Warbler",
    "Muscovy Duck",
    "Mute Swan",
    "Nashville Warbler",
    "Nelsons Sparrow",
    "Neotropic Cormorant",
    "Northern Bobwhite",
    "Northern Cardinal",
    "Northern Flicker",
    "Northern Gannet",
    "Northern Goshawk",
    "Northern Harrier",
    "Northern Hawk Owl",
    "Northern Mockingbird",
    "Northern Parula",
    "Northern Pintail",
    "Northern Rough winged Swallow",
    "Northern Saw whet Owl",
    "Northern Shrike",
    "Northern Waterthrush",
    "Nuttalls Woodpecker",
    "Oak Titmouse",
    "Olive Sparrow",
    "Olive sided Flycatcher",
    "Orange crowned Warbler",
    "Orchard Oriole",
    "Osprey",
    "Ovenbird",
    "Pacific Golden Plover",
    "Pacific Loon",
    "Pacific Wren",
    "Pacific slope Flycatcher",
    "Painted Bunting",
    "Painted Redstart",
    "Palm Warbler",
    "Pectoral Sandpiper",
    "Peregrine Falcon",
    "Phainopepla",
    "Philadelphia Vireo",
    "Pied billed Grebe",
    "Pigeon Guillemot",
    "Pileated Woodpecker",
    "Pine Grosbeak",
    "Pine Siskin",
    "Pine Warbler",
    "Piping Plover",
    "Plumbeous Vireo",
    "Prairie Falcon",
    "Prairie Warbler",
    "Prothonotary Warbler",
    "Purple Finch",
    "Purple Gallinule",
    "Purple Martin",
    "Purple Sandpiper",
    "Pygmy Nuthatch",
    "Pyrrhuloxia",
    "Red Crossbill",
    "Red Knot",
    "Red Phalarope",
    "Red bellied Woodpecker",
    "Red breasted Merganser",
    "Red breasted Nuthatch",
    "Red breasted Sapsucker",
    "Red cockaded Woodpecker",
    "Red eyed Vireo",
    "Red headed Woodpecker",
    "Red naped Sapsucker",
    "Red necked Grebe",
    "Red necked Phalarope",
    "Red shouldered Hawk",
    "Red tailed Hawk",
    "Red throated Loon",
    "Red winged Blackbird",
    "Reddish Egret",
    "Redhead",
    "Ring billed Gull",
    "Ring necked Duck",
    "Ring necked Pheasant",
    "Rock Pigeon",
    "Rock Ptarmigan",
    "Rock Sandpiper",
    "Rock Wren",
    "Rose breasted Grosbeak",
    "Roseate Tern",
    "Rosss Goose",
    "Rough legged Hawk",
    "Royal Tern",
    "Ruby crowned Kinglet",
    "Ruby throated Hummingbird",
    "Ruddy Duck",
    "Ruddy Turnstone",
    "Ruffed Grouse",
    "Rufous Hummingbird",
    "Rufous crowned Sparrow",
    "Rusty Blackbird",
    "Sage Thrasher",
    "Saltmarsh Sparrow",
    "Sanderling",
    "Sandhill Crane",
    "Sandwich Tern",
    "Says Phoebe",
    "Scaled Quail",
    "Scarlet Tanager",
    "Scissor tailed Flycatcher",
    "Scotts Oriole",
    "Seaside Sparrow",
    "Sedge Wren",
    "Semipalmated Plover",
    "Semipalmated Sandpiper",
    "Sharp shinned Hawk",
    "Sharp tailed Grouse",
    "Short billed Dowitcher",
    "Short eared Owl",
    "Snail Kite",
    "Snow Bunting",
    "Snow Goose",
    "Snowy Egret",
    "Snowy Owl",
    "Snowy Plover",
    "Solitary Sandpiper",
    "Song Sparrow",
    "Sooty Grouse",
    "Sora",
    "Spotted Owl",
    "Spotted Sandpiper",
    "Spotted Towhee",
    "Spruce Grouse",
    "Stellers Jay",
    "Stilt Sandpiper",
    "Summer Tanager",
    "Surf Scoter",
    "Surfbird",
    "Swainsons Hawk",
    "Swainsons Thrush",
    "Swallow tailed Kite",
    "Swamp Sparrow",
    "Tennessee Warbler",
    "Thayers Gull",
    "Townsends Solitaire",
    "Townsends Warbler",
    "Tree Swallow",
    "Tricolored Heron",
    "Tropical Kingbird",
    "Trumpeter Swan",
    "Tufted Titmouse",
    "Tundra Swan",
    "Turkey Vulture",
    "Upland Sandpiper",
    "Varied Thrush",
    "Veery",
    "Verdin",
    "Vermilion Flycatcher",
    "Vesper Sparrow",
    "Violet green Swallow",
    "Virginia Rail",
    "Wandering Tattler",
    "Warbling Vireo",
    "Western Bluebird",
    "Western Grebe",
    "Western Gull",
    "Western Kingbird",
    "Western Meadowlark",
    "Western Sandpiper",
    "Western Screech Owl",
    "Western Scrub Jay",
    "Western Tanager",
    "Western Wood Pewee",
    "Whimbrel",
    "White Ibis",
    "White breasted Nuthatch",
    "White crowned Sparrow",
    "White eyed Vireo",
    "White faced Ibis",
    "White headed Woodpecker",
    "White rumped Sandpiper",
    "White tailed Hawk",
    "White tailed Kite",
    "White tailed Ptarmigan",
    "White throated Sparrow",
    "White throated Swift",
    "White winged Crossbill",
    "White winged Dove",
    "White winged Scoter",
    "Wild Turkey",
    "Willet",
    "Williamsons Sapsucker",
    "Willow Flycatcher",
    "Willow Ptarmigan",
    "Wilsons Phalarope",
    "Wilsons Plover",
    "Wilsons Snipe",
    "Wilsons Warbler",
    "Winter Wren",
    "Wood Stork",
    "Wood Thrush",
    "Worm eating Warbler",
    "Wrentit",
    "Yellow Warbler",
    "Yellow bellied Flycatcher",
    "Yellow bellied Sapsucker",
    "Yellow billed Cuckoo",
    "Yellow billed Magpie",
    "Yellow breasted Chat",
    "Yellow crowned Night Heron",
    "Yellow eyed Junco",
    "Yellow headed Blackbird",
    "Yellow rumped Warbler",
    "Yellow throated Vireo",
    "Yellow throated Warbler",
    "Zone tailed Hawk",
]


# Land: soil, earth, terrain, ground, landscape, territory, acreage, property, real estate, plot, parcel, tract, lot, area, region, country, continent, planet.
# Bamboo: cane, reed, stalk, stem, shoot, pole, stick, plant, grass, tree, wood, material, construction, furniture, decoration, craft, art.
# Forest: woodland, jungle, rainforest, grove, copse, thicket, bush, scrub, plantation, timberland, park, reserve, wildlife, habitat, ecosystem, environment.
# Bamboo forest: grove, thicket, plantation, jungle, woodland, rainforest, habitat, ecosystem, environment.
# Broadleaf: deciduous, hardwood, leafy, foliage, tree, plant, forest, woodland, jungle, rainforest, habitat, ecosystem, environment.
# Forest broadleaf: deciduous forest, hardwood forest, leafy forest, broadleaf forest, woodland, jungle, rainforest, habitat, ecosystem, environment.
# Conifer: evergreen, pine, fir, spruce, cedar, hemlock, tree, plant, forest, woodland, jungle, rainforest, habitat, ecosystem, environment.
# Forest conifer: evergreen forest, pine forest, fir forest, spruce forest, cedar forest, hemlock forest, coniferous forest, woodland, jungle, rainforest, habitat, ecosystem, environment.
# Mixed forest: combination forest, blend forest, hybrid forest, forest with both broadleaf and conifer trees, woodland, jungle, rainforest, habitat, ecosystem, environment.
# Grassland: prairie, savanna, meadow, steppe, plain, field, pasture, range, grazing land, habitat, ecosystem, environment.
# Grass: lawn, turf, meadow, pasture, range, hay, straw, fodder, feed, plant, habitat, ecosystem, environment.
# Mountain: peak, summit, range, ridge, hill, slope, cliff, rock, boulder, terrain, landscape, habitat, ecosystem, environment.
# Field: meadow, pasture, range, plain, prairie, savanna, steppe, grassland, crop, farm, agriculture, habitat, ecosystem, environment.

# Ocean: sea, saltwater, deep, blue, waves, current, tide, marine life, coral reef, abyss, trench, pelagic, abyssal, benthic, plankton, whale, shark, dolphin, octopus, jellyfish.
# Lake: freshwater, still, calm, surface, shore, beach, fishing, boating, swimming, recreation, ecosystem, habitat, algae, fish, duck, goose, swan, turtle.
# Water: liquid, H2O, transparent, colorless, tasteless, odorless, essential, hydration, drinking, washing, cleaning, irrigation, hydroelectricity, ice, steam, vapor.
# Sea: ocean, saltwater, coastal, shallow, blue, waves, current, tide, marine life, coral reef, beach, seashell, seagull, crab, lobster, shrimp, oyster.
# River: freshwater, flowing, current, stream, bank, bed, delta, estuary, source, mouth, tributary, rapids, waterfall, fishing, boating, irrigation, hydroelectricity, ecosystem, habitat, fish, otter, beaver.
# Stream: freshwater, flowing, current, creek, brook, rivulet, runnel, channel, bed, source, mouth, fishing, hiking, ecosystem, habitat, fish, frog, turtle.
# Pond: freshwater, still, small, shallow, surface, shore, bank, fishing, boating, swimming, ecosystem, habitat, algae, fish, duck, goose, turtle.
# Marsh: wetland, swamp, bog, fen, mire, quagmire, slough, sedge, reed, cattail, ecosystem, habitat, bird, frog, turtle, mosquito.
# Coast: shoreline, beach, cliff, dune, bay, cove, inlet, estuary, headland, peninsula, lighthouse, fishing, boating, recreation, ecosystem, habitat, seagull, crab, lobster, shrimp, oyster.

LAND_BACK_WORDS = [
    "land",
    "bamboo",
    "forest",
    "bamboo forest",
    "broadleaf",
    "forest broadleaf"
    "conifer",
    "forest conifer",
    "mixed forest",
    "grassland",
    "grass",
    "mountain",
    "field",
    "ground",
    "soil",
    "rainforest",
    "leafy",
    "tree",
    "woodland",
    "jungle",
    "evergreen",
    "plant",
    "deciduous",
]
WATER_BACK_WORDS = [
    "ocean",
    "lake",
    "water",
    "sea",
    "river",
    "stream",
    "pond",
    "marsh",
    "coast",
    "wetland",
    "coastline",
    "shoreline",
    "freshwater",
    "pelagic",
]
AUX_BACK_WORDS = [
    "beach",
    "desert",
    "sky",
    "cloud",
    "snow",
    "ice",
    "rock",
    "sand",
    "mud",
    "dirt",
    "road",
    "swamp",
]

PROMPT_BIRD_NORMAL = [[f.format(v) for f in BIRD_TEMPLATES] for v in WATER_BIRD_WORDS]
PROMPT_BIRD_ANOMALY = [[f.format(v) for f in BIRD_TEMPLATES] for v in LAND_BIRD_WORDS]
PROMPT_BIRD_AUXILIARY = [[f.format(v) for f in BIRD_TEMPLATES] for v in AUX_BIRD_WORDS]

PROMPT_BACK_NORMAL = [[f.format(v) for f in BACK_TEMPLATES] for v in WATER_BACK_WORDS]
PROMPT_BACK_ANOMALY = [[f.format(v) for f in BACK_TEMPLATES] for v in LAND_BACK_WORDS]
PROMPT_BACK_AUXILIARY = [[f.format(v) for f in BACK_TEMPLATES] for v in AUX_BACK_WORDS]


def get_labels(attrs, guidance: str):
    if guidance == "guide_bird" or guidance == "ignore_back":
        attend_name, ignore_name = "bird", "back"
        attend_labels, ignore_labels = attrs[:, 0], attrs[:, 1]
    elif guidance == "guide_back" or guidance == "ignore_bird":
        attend_name, ignore_name = "back", "bird"
        attend_labels, ignore_labels = attrs[:, 1], attrs[:, 0]
    else:
        raise ValueError(f"Invalid guidance: {guidance}")

    return attend_name, ignore_name, attend_labels, ignore_labels


def get_prompts(guidance: str):
    if guidance == "guide_bird" or guidance == "ignore_bird":
        prompts = {
            "normal": PROMPT_BIRD_NORMAL,
            "anomaly": PROMPT_BIRD_ANOMALY + PROMPT_BIRD_AUXILIARY,
            "half": PROMPT_BIRD_NORMAL + PROMPT_BIRD_ANOMALY[:len(PROMPT_BIRD_ANOMALY) // 2],
            "exact": PROMPT_BIRD_NORMAL + PROMPT_BIRD_ANOMALY,
            "all": PROMPT_BIRD_NORMAL + PROMPT_BIRD_ANOMALY + PROMPT_BIRD_AUXILIARY,
        }
    elif guidance == "guide_back" or guidance == "ignore_back":
        prompts = {
            "normal": PROMPT_BACK_NORMAL,
            "anomaly": PROMPT_BACK_ANOMALY + PROMPT_BACK_AUXILIARY,
            "half": PROMPT_BACK_NORMAL + PROMPT_BACK_ANOMALY[:len(PROMPT_BACK_ANOMALY) // 2],
            "exact": PROMPT_BACK_NORMAL + PROMPT_BACK_ANOMALY,
            "all": PROMPT_BACK_NORMAL + PROMPT_BACK_ANOMALY + PROMPT_BACK_AUXILIARY,
        }
    else:
        raise ValueError(f"Invalid guidance: {guidance}")

    return prompts


def get_words(guidance: str):
    if guidance == "guide_bird" or guidance == "ignore_bird":
        prompts = {
            "normal": WATER_BIRD_WORDS,
            "anomaly": LAND_BIRD_WORDS + AUX_BIRD_WORDS,
            "half": WATER_BIRD_WORDS + LAND_BIRD_WORDS[:len(LAND_BIRD_WORDS) // 2],
            "exact": WATER_BIRD_WORDS + LAND_BIRD_WORDS,
            "all": WATER_BIRD_WORDS + LAND_BIRD_WORDS + AUX_BIRD_WORDS,
        }
    elif guidance == "guide_back" or guidance == "ignore_back":
        prompts = {
            "normal": WATER_BACK_WORDS,
            "anomaly": LAND_BACK_WORDS + AUX_BACK_WORDS,
            "half": WATER_BACK_WORDS + LAND_BACK_WORDS[:len(LAND_BACK_WORDS) // 2],
            "exact": WATER_BACK_WORDS + LAND_BACK_WORDS,
            "all": WATER_BACK_WORDS + LAND_BACK_WORDS + AUX_BACK_WORDS,
        }
    else:
        raise ValueError(f"Invalid guidance: {guidance}")

    if guidance.startswith("ignore"):
        prompts = {k: [v for w in prompts[k] for v in w] for k in prompts}

    return prompts
