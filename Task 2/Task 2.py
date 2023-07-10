import pickle
import os
import io
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import sys
from annoy import AnnoyIndex
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_SIZE = 224

# Get full working directory
cwd = os.getcwd()
print(cwd)

df_path = os.path.join(cwd, 'df.pickle')

# Import the data from pickles
df = pickle.load(open(df_path, 'rb'))
feature_list = pickle.load(open(os.path.join(cwd, 'features.pickle'), 'rb'))


def convert_to_jpg_arr(image_path):
    with Image.open(image_path) as image:
        with io.BytesIO() as output:
            image.convert('RGB').save(output, format='JPEG')
            output.seek(0)
            return np.asarray(Image.open(output))

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def extract_features(img_path, model, image_size=IMAGE_SIZE):
# Create a new model that outputs the desired feature layer
    print(img_path) # For debugging

    feature_model = torch.nn.Sequential(*list(model.children())[:-2])
    feature_model.eval()

    # Load and preprocess the image
    image = Image.open(img_path)
    image = image.convert('RGB')

    resize = transform_test(image).to(device)
    
    print(resize.shape)

    # Extract features from the image
    features = feature_model(resize.unsqueeze(0))

    # Normalize the features
    flattened_features = features.flatten()
    normalised_features = flattened_features / torch.norm(flattened_features)

    # Convert to NumPy Array
    normalised_features = normalised_features.cpu().detach().numpy()

    return normalised_features

# Load the model
if torch.cuda.is_available():
    model = torch.load(os.path.join(cwd,'model.pth'))
else:
    model = torch.load(os.path.join(cwd,'model.pth'), map_location=torch.device('cpu'))

image = sys.argv[1]

features = extract_features(image, model)

# Add new_features to the feature_list2
feature_list2 = feature_list.copy()
feature_list2.append(features)

# Clone df and add the new image to it
df2 = df.copy()
df2.loc[len(df2)] = [image, features, '']

# Create annoyindex tree
f = len(df2['img_repr'][0])
t = AnnoyIndex(f, metric='euclidean')

for i in tqdm(range(len(feature_list2))):
    t.add_item(i, feature_list2[i])
    
_ = t.build(150, n_jobs=-1)

def get_similar_images_annoy(img_index):
    base_img_id, base_vector, base_label  = df2.iloc[img_index, [0, 1, 2]]
    similar_img_ids = t.get_nns_by_item(img_index, 11)
    return base_img_id, base_label, df2.iloc[similar_img_ids[1:]]

base_image, base_label, similar_images_df = get_similar_images_annoy(len(df2)-1)

print(similar_images_df)

def save_images(new_img_path, output_file):
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 20))
    
    # Plot the base image
    axes[0, 0].imshow(convert_to_jpg_arr(new_img_path))
    axes[0, 0].set_title('Base Image')
    axes[0, 0].axis('off')
    
    # Plot similar images
    for i in range(len(similar_images_df)):
        path = os.path.join(similar_images_df.iloc[i, 0])
        image = mpimg.imread(path)
        axes[(i + 1) // 4, (i + 1) % 4].imshow(image)
        axes[(i + 1) // 4, (i + 1) % 4].axis('off')

    # Adjust the layout and save the plot to a file
    plt.tight_layout()
    plt.savefig(output_file)

save_images(base_image, 'reverse_search_output.png')