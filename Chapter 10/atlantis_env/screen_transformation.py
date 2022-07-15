import random
import gym
import matplotlib.pyplot as plt
from ch10.utils import image_rgb_to_grayscale

env = gym.make('Atlantis-v0')
action_size = env.action_space.n

# Generating Random Screen
env.reset()
for _ in range(1000):
    img, reward, done, _ = env.step(random.randint(0, action_size - 1))

# Displaying RGB Full Game Screen
print(f'RGB image: {img.shape}')
plt.title("RGB Full Screen")
plt.xticks([])
plt.yticks([])
plt.imshow(img)
plt.show()

# Displaying Grayscale Full Game Screen
gs_img = image_rgb_to_grayscale(img)
print(f'Gray Scale image: {gs_img.shape}')
plt.title("Grayscale Full Screen")
plt.xticks([])
plt.yticks([])
plt.imshow(gs_img, cmap = plt.get_cmap("gray"))
plt.show()

# Displaying Cropped Grayscale Game Screen
gs_img_cropped = image_rgb_to_grayscale(img, crop_down = 80)
plt.title("Grayscale Cropped Screen")
print(f'Gray Scale image Cropped: {gs_img_cropped.shape}')
plt.xticks([])
plt.yticks([])
plt.imshow(gs_img_cropped, cmap = plt.get_cmap("gray"))
plt.show()
