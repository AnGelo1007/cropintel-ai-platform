CROP AI DATASET BOOST PACK
=================================

What this pack does
-------------------
This pack helps you improve your crop classifier using your existing project structure:

crop_ai_platform/
  dataset_crops/
    corn/
    eggplant/
    pepper/
    rice/
    tomato/

Files included
--------------
1. prepare_dataset.py
   - Splits your crop folders into train/val/test
   - Creates:
       dataset_split/
         train/
         val/
         test/

2. augment_to_target.py
   - If a class has fewer than the target number of images, it creates augmented copies
   - Good for making eggplant and pepper reach at least 100 images each

3. TRAINING_STEPS.txt
   - Step-by-step commands to run

Recommended workflow
--------------------
A. Put new pepper images into:
   dataset_crops/pepper/

B. Put new eggplant images into:
   dataset_crops/eggplant/

C. Run:
   python augment_to_target.py --input dataset_crops --target 100

D. Run:
   python prepare_dataset.py --input dataset_crops --output dataset_split

E. Train:
   python train_crop_classifier.py

Important
---------
- This pack assumes each folder under dataset_crops is a class folder.
- If your train_crop_classifier.py already expects a different folder structure, update the paths in that file.
- Better accuracy usually comes from:
  * more real images
  * balanced class counts
  * cleaner labels
  * train/val/test split
  * image augmentation
