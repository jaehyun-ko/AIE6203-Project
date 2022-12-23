import os
import torchvision.transforms as transforms



def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        

# 논문에 제시된 전처리 구현
class Transforms:
    def to_gray(frame1, frame2=None):
        gray_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.CenterCrop((175,150)),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

        # 공과 bar를 추적하기 위한 frame extraction
        if frame2 is not None:
            new_frame = gray_transform(frame2) - 0.4*gray_transform(frame1)
        else:
            new_frame = gray_transform(frame1)

        return new_frame.numpy()