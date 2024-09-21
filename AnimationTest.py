# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 21:58:55 2024

@author: tarun
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib
nx = 50

fig, ax = plt.subplots()
data = np.random.rand(3, nx)
t = np.ones((3, nx)) * np.arange(nx)
#data[:, :, 0].fill(0.2)
plt.xlim(-1, 51)
plt.ylim(0, 1)
im = ax.scatter(t[:, 0], data[:, 0])
label = fig.text(0.1, 0.1, "Image 0", fontsize=10)

plt.title("Data")
plt.xlabel("X")
plt.ylabel("Y")
'''
def init():
    im.set_data(data[:, :, 0])
    label.set_text("Image" + str(0))
'''
def animate(i):
    im.set_offsets((t[:, :i], data[:, :i]))
    label.set_text("Image" + str(i))
    return im

anim = animation.FuncAnimation(fig, animate, frames=data.shape[0], repeat = False)

plt.show()
plt.rcParams['animation.ffmpeg_path'] = "D:\\ffmpeg-7.0.2-essentials_build\\ffmpeg-7.0.2-essentials_build\\bin\\ffmpeg.exe"
writer = animation.FFMpegWriter(fps=0.3)
anim.save('filename.mkv', writer=writer)
#anim.save('filename.mp4', writer='ffmpeg', fps=0.3)
#anim.save('sine.mp3', dpi=150, fps = 30, writer='Pillow')