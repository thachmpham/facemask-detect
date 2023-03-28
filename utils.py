import matplotlib.pyplot as plt
import PIL

def show_side_by_side(path_a, path_b):
    img_a = PIL.Image.open(path_a)
    img_b = PIL.Image.open(path_b)
    
    fig, ax = plt.subplots(1,2)
    
    ax[0].imshow(img_a)
    ax[1].imshow(img_b)