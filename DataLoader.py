import os
import numpy as np
import cv2

class ImageFolder:
    def __init__(self, folderPath):
        self.path = folderPath
        self.images = []

    def loadImagesToList(folder):
        images = []
        fileNames = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                fileNames.append(filename)
                images.append(img)
        return filename, images





def main():


if __name__ == "__main__":
    main()
