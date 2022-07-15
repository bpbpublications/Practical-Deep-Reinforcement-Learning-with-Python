import random
import gym
import matplotlib.pyplot as plt
from ch10.dqn.screen_motion import ScreenMotion
from ch10.utils import image_rgb_to_grayscale

env = gym.make('Atlantis-v0')
action_size = env.action_space.n

motion = ScreenMotion()

# Executing 100 random actions in Atlantis-v0
env.reset()
for i in range(100):
    img, reward, done, _ = env.step(random.randint(0, action_size - 1))
    gs_img_cropped = image_rgb_to_grayscale(img, crop_down = 80)
    motion.add(gs_img_cropped)

# Displaying Motion
frames = motion.get_frames()
for i, f in enumerate(frames):
    plt.title(f"Frame: {i}")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(f, cmap = plt.get_cmap("gray"))
    plt.show()
