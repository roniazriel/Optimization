import pandas as pd
import matplotlib.pyplot as plt
from tab2img.converter import Tab2Img


''' Convert data to image for CNN'''
dataset = pd.read_csv("grouped_data_6dof_old.csv")
dataset = dataset.drop("Unnamed: 0", axis=1)
print(dataset)
print(dataset.keys())

'''Convert to image using 1 target'''
target1 = dataset[['Success_Rates']]
target2 = dataset[['Manipulability_Rates']]
dataset.drop(['Success_Rates','Manipulability_Rates'], inplace=True, axis=1)

model1 = Tab2Img()
images1 = model1.fit_transform(dataset.to_numpy(), target2.to_numpy())
fig, axes = plt.subplots(10,10, figsize=(8,8))

# plot 100 images (10X10)
for i,ax in enumerate(axes.flat):
    ax.imshow(images1[i])
    print(images1[i].shape)
fig.suptitle('Images produced using Success rate as a target', fontsize=14)
plt.show()

model2 = Tab2Img()
images2 = model2.fit_transform(dataset.to_numpy(), target2.to_numpy())
fig, axes = plt.subplots(10,10, figsize=(8,8))

# plot 100 images (10X10)
for i,ax in enumerate(axes.flat):
    ax.imshow(images2[i])
fig.suptitle('Images produced using Success rate as a target', fontsize=14)
plt.show()

