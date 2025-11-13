import albumentations as A

def transform(size):
    transforms = A.Compose([
        A.Resize(size, size),
        A.Normalize()
        ])
    return transforms