import torch.utils.data as data

from PIL import Image

import os
import os.path
import numpy as np

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    if isinstance(dir, list):
        dir = dir[0]
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(class_to_idx, extensions, dirs=None, image_path=None):
    images = []
    if image_path is None and dirs is not None:
        for dir in dirs:
            dir = os.path.expanduser(dir)
            for target in sorted(os.listdir(dir)):
                d = os.path.join(dir, target)
                if not os.path.isdir(d):
                    continue

                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        if has_file_allowed_extension(fname, extensions):
                            path = os.path.join(root, fname)
                            item = (path, class_to_idx[target])
                            images.append(item)
    else:
        for image in image_path:
            if has_file_allowed_extension(image, extensions):
                target = os.path.split(os.path.split(image)[0])[1]
                item = (image, class_to_idx[target])
                images.append(item)
            
    return images


class DatasetFolder(data.Dataset):

    def __init__(self, loader, extensions, root=None, image_path=None, transform=None, target_transform=None):
        if root is None and image_path is None:
            raise(RuntimeError("root or image_path must be given"))        
        if root is not None:
            classes, class_to_idx = find_classes(root)
        else:
            root_t = os.path.split(os.path.split(image_path[0])[0])[0]
            classes, class_to_idx = find_classes(root_t)           
        samples = make_dataset(class_to_idx, extensions, dirs=root, image_path=image_path)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(image=np.array(sample))['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    def __init__(self, root=None, image_path=None, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(loader=default_loader,
                                          extensions=IMG_EXTENSIONS,
                                          root=root,
                                          image_path=image_path,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
