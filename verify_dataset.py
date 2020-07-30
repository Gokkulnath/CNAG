from glob import glob

train_ok = True
val_ok = True
print("Training Data Verification")
cls_count = len(glob("ILSVRC/train/*"))
print("Total Number of Classes: {} in train directory".format(cls_count))
count = 0
for cls_ in glob("ILSVRC/train/*"):
    imgs = glob(cls_ + "/*")
    img_count = len(imgs)
    count += img_count
    if img_count != 10:
        print(cls_.split("/")[-1], img_count)
        train_ok=False
print("Total {} number of files in {} classes. i.e 10 Images/Class".format(count, cls_count))

print("Validation Data Verification")
val_files = glob("ILSVRC/valid/*")
val_count = len(val_files)
if val_count == 50000:
    print("Validation Data has correct number of files i.e {}".format(val_count))
else:
    print("Validation Data has some issue. Has following number of file : {}. Kindly Check!!".format(val_count))
    val_ok=False
if train_ok and val_ok:
    print("Dataset is Setup Correctly")