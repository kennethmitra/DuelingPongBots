import imageio

class GIF_Recorder:

    def __init__(self, save_dir):
        self.images = []
        self.save_dir = save_dir

    def add_image(self, image):
        self.images.append(image)

    def make_gif(self, epoch):
        imageio.mimsave(f"{self.save_dir}/{epoch}.gif", self.images)
        self.images = []