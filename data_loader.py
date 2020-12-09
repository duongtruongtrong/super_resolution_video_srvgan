import cv2
import tensorflow as tf
import random

class DataLoader():
    """Data Loader for the SR GAN, that prepares a tf data object for training.
    Functions to use:
        train_dataset(image_paths)
        val_dataset(image_paths)
    """

    def __init__(self, hr_height, hr_width, lr_height, lr_width, batch_size):
        """
        Initializes the dataloader.
        Args:
            hr_height: Int. Height of desired output high resolution image.
            hr_width: Int. Width of desired output high resolution image.
            lr_height: Int. Height of desired output low resolution image.
            lr_width: Int. Width of desired output low resolution image.
            batch_size: Int. Size of training and validation batch in tf.dataset.
        Returns:
            The dataloader object.
        """
        self.hr_height = hr_height
        self.hr_width = hr_width

        self.lr_height = lr_height
        self.lr_width = lr_width

        self.batch_size = batch_size

    # Training Dataset

    # 1 image as 1 element

    def _reverse(self, ds):
        """
        Function that randomly reverse frames sequence in 1 video.
        Args:
            ds: A tf dataset.
        Returns:
            ds: A tf dataset with reversed frames sequence.
        """ 
    #     reverse squence randomly
        method_list = ['reverse', None]
        reverse_method = random.choice(method_list)
        
        image_list = list(ds.as_numpy_iterator())

        if reverse_method == 'reverse':
            image_list.reverse()

        return tf.data.Dataset.from_tensor_slices(image_list)

    def _parse_image(self, image_path):
        """
        Function that loads the images given the path.
        Args:
            image_path: The paths to frames in the video.
        Returns:
            image: A tf tensor of the loaded frames.
        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image

    def _random_crop_resize(self, ds):
        """
        Function that randomly crop or resize image to desired resolution to produce high_res image.
        Args:
            ds: A tf dataset.
        Returns:
            ds: A tf dataset with cropped or resized frames.
        """
        method_list = ['crop', 'resize']
        crop_resize_method = random.choice(method_list)

        method_list = ['bilinear', 'gaussian', 'nearest', 'area']
        downsampling_method = random.choice(method_list)

        def random_crop(image):
            """
            Function that randomly crop image to desired resolution to produce high_res image.
            Args:
                image: A tf tensor of the loaded frames.
            Returns:
                image: A tf tensor of cropped frames.
            """

            # resolution under 360 is too smal that cropping out too much details of the image.
            # if hr_height < 360:
            image = tf.image.random_crop(image, [360, 640, 3])
            image = tf.image.resize(image, 
                                    [self.hr_height, self.hr_width],
                                    preserve_aspect_ratio=True,
                                    method=downsampling_method)

            return image

        def downsampling(image):
            """
            Function that resize image to desired resolution.
            Downsampling methods: ['bilinear', 'gaussian', 'nearest', 'area']
            Args:
                image: A tf tensor of the loaded frames.
            Returns:
                image: A tf tensor of resized frames.
            """
    #         print(tf.shape(high_res)[0])
            image = tf.image.resize(image, 
                                    [self.hr_height, self.hr_width],
                                    preserve_aspect_ratio=True,
                                    method=downsampling_method)

            return image

        if crop_resize_method == 'crop':
            # randomly crop frame
            ds = ds.map(random_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            # downsampling frame
            ds = ds.map(downsampling, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return ds

    def _flip(self, ds):
        """
        Function that flip horizontally/vertically all images in 1 dataset.
        Args:
            ds: A tf dataset.
        Returns:
            ds: A tf dataset with flipped images.
        """ 
    #     flip the image randomly
        method_list = ['horizontal', 'vertical', None]
        flip_method = random.choice(method_list)
        
        def flip_left_right(image):
            image = tf.image.flip_left_right(image)
            return image
        
        def flip_up_down(image):
            image = tf.image.flip_up_down(image)
            return image
        
        if flip_method == 'horizontal':
            ds = ds.map(flip_up_down, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        elif flip_method == 'vertical':
            ds = ds.map(flip_left_right, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            
        return ds

    def _high_low_res_pairs(self, ds):
        """
        Function that generates a low resolution image given the high resolution image with random methods.
        Downsampling methods: ['bilinear', 'gaussian', 'nearest', 'area']
        Args:
            ds: A tf dataset.
        Returns:
            ds: A tf dataset with low and high res images.
        """
        method_list = ['bilinear', 'gaussian', 'nearest', 'area']
        downsampling_method = random.choice(method_list)

        def downsampling(high_res):
            """
            Function that generates a low resolution image given the high resolution image.
            Args:
                high_res: A tf tensor of the high res image.
            Returns:
                low_res: A tf tensor of the low res image.
                high_res: A tf tensor of the high res image.
            """
    #         print(tf.shape(high_res)[0])
            low_res = tf.image.resize(high_res, 
                                    [self.lr_height, self.lr_width],
                                    preserve_aspect_ratio=True,
                                    method=downsampling_method)

            return low_res, high_res
        
        ds = ds.map(downsampling, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        return ds

    def _rescale(self, low_res, high_res):
        """
        Function that rescales the pixel values of high_res to the -1 to 1 range.
        For use with the generator output tanh function.
        Args:
            low_res: The tf tensor of the low res image.
            high_res: The tf tensor of the high res image.
        Returns:
            low_res: The tf tensor of the low res image, rescaled.
            high_res: The tf tensor of the high res image, rescaled.
        """
        high_res = high_res * 2.0 - 1.0

        return low_res, high_res

    def train_dataset(self, image_paths):
        """
        Returns a tf dataset object with specified mappings. No shuffle and No repeat.
        No shuffle because it will screw up the frame sequence.
        No repeat because training model will use a manual for loop.
        Args:
            image_paths: Str, Path to images.
            batch_size: Int, The number of elements in a batch returned by the dataset.
        Returns:
            dataset: A tf dataset object.
        """
        
        # Generate tf dataset from high res video paths.
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        # Prefetch the data for optimal GPU utilization.
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        # apply: Applies a transformation function to the whole dataset as once. Good for functions with the same random arg.

        # randomly reverse frames sequence in 1 video
        dataset = dataset.apply(self._reverse)

        # image paths to tensor
        dataset = dataset.map(self._parse_image, num_parallel_calls=AUTOTUNE)

        # randomly crop frame
        dataset = dataset.apply(self._random_crop_resize)

        # randomly flip all frames in 1 video
        dataset = dataset.apply(self._flip)

        # Generate low resolution by downsampling.
        dataset = dataset.apply(self._high_low_res_pairs)

        # Rescale the values in the input
        dataset = dataset.map(self._rescale, num_parallel_calls=AUTOTUNE)

        # Batch the input, drop remainder to get a defined batch size.
        dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(AUTOTUNE)

        return dataset

    # Validation Dataset

    # 1 image as 1 element
    def _val_low_res(self, ds):
        """
        Function that generates a low resolution image given the high resolution image with random methods.
        Listed methods: ['bilinear',, 'gaussian', 'nearest', 'area']
        Args:
            ds: A tf dataset.
        Returns:
            ds: A tf dataset with low and high res images.
        """
        method_list = ['bilinear', 'gaussian', 'nearest', 'area']
        downsampling_method = random.choice(method_list)
        
        def downsampling(high_res):
            """
            Function that generates a low resolution image given the high resolution image.
            Args:
                high_res: A tf tensor of the high res image.
            Returns:
                low_res: A tf tensor of the low res image.
                high_res: A tf tensor of the high res image.
            """

            low_res = tf.image.resize(high_res, 
                                    [self.lr_height, self.lr_width],
                                    preserve_aspect_ratio=True,
                                    method=downsampling_method)   
            return low_res
        
        ds = ds.map(downsampling, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        return ds

    def val_dataset(self, image_paths):
        """
        Returns a tf dataset object with specified mappings. No shuffle and repeat.
        Args:
            image_paths: Str, Path to images.
            batch_size: Int, The number of elements in a batch returned by the dataset.
        Returns:
            dataset: A tf dataset object.
        """
        
        # Generate tf dataset from high res video paths.
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)

        # Prefetch the data for optimal GPU utilization.
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        # image paths to tensor
        dataset = dataset.map(self._parse_image, num_parallel_calls=AUTOTUNE)
        
        # Generate low resolution by downsampling.
        dataset = dataset.apply(self._val_low_res)

        # Batch the input, drop remainder to get a defined batch size.
        dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(AUTOTUNE)

        return dataset