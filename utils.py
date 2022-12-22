import os
import torchvision.transforms as transforms



def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        
        


# 중앙을 자르고 resize
class Transforms:
    def to_gray(frame1, frame2=None):
        gray_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.CenterCrop((175,150)),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

        # diff frame으로 성능 개선
        if frame2 is not None:
            new_frame = gray_transform(frame2) - 0.4*gray_transform(frame1)
        else:
            new_frame = gray_transform(frame1)

        return new_frame.numpy()