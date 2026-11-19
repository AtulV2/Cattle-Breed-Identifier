# 🐄 Dataset & 50 Cattle Breed Catalog

This document provides a comprehensive overview of the **Indian Cattle Image Dataset**, the importance of class mapping determinism, and a detailed reference table covering all **50 recognized cattle breeds**.

---

## 1. Dataset Overview

- **Source:** [Indian Cattle Image Dataset on Kaggle](https://www.kaggle.com/datasets/atharvadarpude/indian-cattle-image-dataset) (Atharva Darpude).
- **Target Livestock Species:** Cattle (*Bos indicus* / Zebu and *Bos taurus* / Exotic).
- **Total Classes:** 50 distinct breeds.
- **Agricultural Significance:** India possesses some of the world's richest genetic diversity in indigenous cattle. These breeds exhibit natural thermo-tolerance, tick/parasite resistance, and resilience to tropical diseases, making their identification and preservation critical for livestock biodiversity.

---

## 2. Canonical Class Mapping (`data/breeds.json`)

MobileNetV2 outputs a 50-dimensional Softmax probability vector:
$$\mathbf{y} = [p_0, p_1, p_2, \dots, p_{49}], \quad \sum_{i=0}^{49} p_i = 1$$

To determine the predicted breed name, the index $k = \arg\max(\mathbf{y})$ is mapped to `TARGET_BREEDS[k]`.

> [!IMPORTANT]
> Filesystem directory listings (`os.listdir`) sort differently across operating systems (e.g. Windows NTFS vs. Linux ext4 vs. macOS APFS case-sensitivity). 
> To guarantee **100% deterministic class indexing across all platforms and cloud environments**, the canonical 50-breed list is stored in [data/breeds.json](file:///D:/Project/Livestock_Breed_ID/data/breeds.json).

---

## 3. Comprehensive 50-Breed Catalog

The 50 recognized breeds span dairy (milch), heavy draught (working), dual-purpose, and dwarf conservation breeds:

| # | Breed Name | Native Region | Category | Key Morphological Traits |
|:---|:---|:---|:---|:---|
| 1 | **Amritmahal** | Karnataka | Draught | Grey/white coat, prominent backward-pointing sharp horns, energetic posture. |
| 2 | **Ayrshire** | Scotland (Exotic) | Dairy | Reddish-brown and white patched coat, lyre-shaped horns, high milk volume. |
| 3 | **Bargur** | Tamil Nadu | Draught | Brown coat with distinct white markings, long slender horns, bred in Bargur hills. |
| 4 | **Dangi** | Maharashtra / Gujarat | Draught | Distinctive white coat with irregular black or red speckles, oily skin secretion. |
| 5 | **Deoni** | Maharashtra / Karnataka | Dual-Purpose | White coat mottled with black patches, drooping ears, prominent forehead. |
| 6 | **Gir** | Gujarat (Saurashtra) | Dairy | Convex broad forehead, pendulous long ears, speckled red/white coat, large hump. |
| 7 | **Hallikar** | Karnataka | Draught | Dark grey coat, sharp vertical backward curving horns, agile working build. |
| 8 | **Hariana** | Haryana / Punjab | Dual-Purpose | Light grey to white coat, compact muscular body, black tail switch. |
| 9 | **Himachali Pahari** | Himachal Pradesh | Draught / Small | Small compact body, dark brown to black coat, adapted to steep mountainous terrain. |
| 10 | **Kangayam** | Tamil Nadu | Draught | Grey/white coat, dark markings on shoulders, strong short neck, hardy hooves. |
| 11 | **Kankrej** | Gujarat / Rajasthan | Dual-Purpose | Heavy lyre-shaped horns, pendulous ears, distinct rhythmic gait ("Savai chal"). |
| 12 | **Kenkatha** | Bundelkhand (UP / MP) | Draught | Small sturdy frame, greyish-brown color, curved horns pointing outwards. |
| 13 | **Khariar** | Odisha | Draught / Milk | Compact frame, brown to grey coat, native to drought-prone Kalahandi region. |
| 14 | **Khillari** | Maharashtra / Karnataka | Draught | Grey-white coat, long narrow face, pink skin around eyes, fast working bullock. |
| 15 | **Konkan Kapila** | Coastal Maharashtra / Goa | Dual-Purpose | Small to medium stature, reddish to grey coat, adapted to heavy coastal rainfall. |
| 16 | **Kosali** | Chhattisgarh | Draught | Small light body, grey/light brown, highly adapted to paddy field working conditions. |
| 17 | **Krishna Valley** | Karnataka / Maharashtra | Heavy Draught | Massive frame, prominent hump, white/grey coat, bred for heavy agricultural tillage. |
| 18 | **Ladakhi** | Ladakh | High Altitude | Short compact body, thick hairy black coat, thrives at altitudes above 3,000m. |
| 19 | **Lakhimi** | Assam | Small / Milk | Small dual breed of Brahmaputra valley, brownish-grey, tolerant to high humidity. |
| 20 | **Malnad Gidda** | Karnataka | Dwarf / Dairy | Miniaturized stature (dwarf), black or brown coat, extremely disease resistant. |
| 21 | **Mewati** | Haryana / Rajasthan | Dual-Purpose | White/light grey coat, docile temperament, resembling Hariana but with Gir influence. |
| 22 | **Nari** | Rajasthan / Gujarat | Draught | White to light brown coat, long horns, adapted to semi-arid desert border areas. |
| 23 | **Nimari** | Madhya Pradesh | Draught | Red coat with irregular white splashes (resembling Gir and Khillari cross). |
| 24 | **Ongole** | Andhra Pradesh | Heavy Draught / Dual | Massive white body, muscular build, dark eyes, globally exported as foundation beef stock. |
| 25 | **Poda Thirupu** | Telangana | Draught | Greyish white with brown or black specks, compact and agile hill draft animal. |
| 26 | **Pulikulam** | Tamil Nadu | Draught / Sport | Dark grey to black markings on head and neck, fierce agility, used in Jallikattu. |
| 27 | **Punganur** | Andhra Pradesh | Dwarf Dairy | Shortest humped cattle breed in the world, broad forehead, docile demeanor. |
| 28 | **Purnea** | Bihar | Draught / Milk | Small light build, reddish-brown to light grey, adapted to North Bihar floodplains. |
| 29 | **Rathi** | Rajasthan (Thar Desert) | Dairy | Brown and white patched coat, medium stature, exceptional heat and drought endurance. |
| 30 | **Red Kandhari** | Maharashtra | Draught | Deep uniform red or dark mahogany coat, medium body, black muzzle and hooves. |
| 31 | **Red Sindhi** | Sindh / Across India | Dairy | Deep rich reddish-brown coat, large hump, heavy dewlap, high milk yield. |
| 32 | **Sahiwal** | Punjab / Haryana | Dairy | Reddish-dun coat, loose skin ("Lola"), large pendulous udder, premier zebu milch breed. |
| 33 | **Shweta Kapila** | Goa | Sacred / Milk | Completely pure white coat, white hooves, white eyelashes, sacred native breed. |
| 34 | **Tharparkar** | Rajasthan / Sindh | Dual-Purpose | Pure white to light grey coat, medium frame, thrives on desert scrub foliage. |
| 35 | **Umblachery** | Tamil Nadu | Draught | Grey coat with white markings on face and legs, prominent star on forehead. |
| 36 | **Vechur** | Kerala | Dwarf Dairy | World's smallest cattle breed, light build, extremely high fat-globule milk. |
| 37 | **Bachaur** | Bihar | Draught | Compact white coat, small straight horns, sturdy bullocks for ploughing. |
| 38 | **Badri** | Uttarakhand | Hill Dairy | Small agile frame, black, brown or grey coat, grazes on Himalayan medicinal shrubs. |
| 39 | **Bhelai (Belahi)** | Haryana / Chandigarh | Migratory Draught | Reddish-brown with white face markings, adapted to sub-Himalayan migratory grazing. |
| 40 | **Dagri** | Gujarat | Draught | Small compact frame, light brown coat, native to tribal forested tracts. |
| 41 | **Gangatari** | Uttar Pradesh / Bihar | Dual-Purpose | White/grey coat, docile, native to the banks of the sacred Ganges river. |
| 42 | **Gaolao** | Maharashtra / MP | Dual-Purpose | White coat with black markings, convex head, high milk and draught capability. |
| 43 | **Ghumsari** | Odisha | Draught | Small stature, light brown, suited to hilly jungle terrain of southern Odisha. |
| 44 | **Kherigarh** | Uttar Pradesh | Draught | White coat with narrow long face, lyre-shaped horns, fast road draught animal. |
| 45 | **Malvi** | Madhya Pradesh | Heavy Draught | Silver-grey coat, massive muscular shoulders, lyre horns, bred for road transport. |
| 46 | **Motu** | Odisha / AP / CG | Dwarf Draught | Very small stature, brown or grey, bred by tribal communities for swampy paddy fields. |
| 47 | **Nagori** | Rajasthan | Draught | Tall, flat forehead, white coat, recognized as India's finest trotting draft breed. |
| 48 | **Ponwar** | Uttar Pradesh | Draught | Black and white mottled coat, aggressive temperament, small sharp horns. |
| 49 | **Siri** | Sikkim / West Bengal | Mountain Draught | Black and white spotted, thick coarse coat, adapted to high altitude cold and snow. |
| 50 | **Thutho** | Nagaland | Semi-Wild / Meat | Sturdy muscular build, black coat, short horns, raised in community forest pastures. |

---

## 4. Class Imbalance & Data Augmentation Strategy

Rare indigenous breeds often have far fewer public photographic samples than popular dairy breeds like *Gir* or *Sahiwal*.

### Augmentation Pipeline
To prevent the model from overfitting to frequent classes while maintaining morphological realism:
1. **Horizontal Flipping ($p=0.5$):** Cattle anatomy is bilateral; horizontal flipping effectively doubles sample variety without distorting breed characteristics.
2. **Constrained Rotation ($\pm 18^\circ$):** Simulates natural terrain slope and camera tilt while keeping animals upright.
3. **Padded Zoom & Crop (80% to 98%):** Simulates varying camera distances in the pasture without clipping horns or hooves.
4. **Photometric Jitter:** Random brightness ($\pm 15\%$) and contrast adjustments ($\pm 15\%$) to accommodate diverse lighting (bright sunlight vs. shaded cowsheds).
