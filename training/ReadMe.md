The training part depends on the type of dataset available.
For collecting dataset:
(All the functions are in create_dataset.py, we need to run the functions as required.)
1. Collecting negative images. For this use store_negative_images()
2. Deleting bad ones .  use delet_bad_ones() But before that you need to create a folder uglies and put some template images of bad ones.
3. The most important one. Create list of background images. Use  create_list to get a bg.txt file.
4. If your positive image is not resized we need to resize it to 50x50. Use resize_positive_image.




Now its time for training.
Run train.sh. But before that, we need to keep in mind the number of negative images available. Run and you will get cascade.xml in data folder. Copy it to Tracking folder and rename it with a label.
