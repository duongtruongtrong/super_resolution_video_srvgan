import tensorflow as tf
from tensorflow import keras

# # 4. Training

class Train():
    def __init__(self, gen_model, disc_model, hr_shape, learning_rate=1e-3, gen_model_save_path='models/generator_upscale_2_times.h5', disc_model_save_path='models/discriminator_upscale_2_times.h5'):
        """disc_patch: output shape of last layer of discriminator model.
        hr_shape: height resolution shape.
        """
        self.gen_model = gen_model
        self.disc_model = disc_model

        self.disc_patch = disc_model.get_layer('disc_output_layer').output_shape[1:]
        self.pretrain_iteration = 1
        self.pretrain_iteration = 1
    
        self.hr_shape = hr_shape

        self.gen_model_save_path = gen_model_save_path
        self.disc_model_save_path = disc_model_save_path
        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        # Get the vgg network. Extract features from Block 5, last convolution, exclude layer block5_pool (MaxPooling2D)
        self.vgg = keras.applications.VGG19(weights="imagenet", input_shape=self.hr_shape, include_top=False)
        self.vgg.trainable = False

        # Create model and compile vgg19 for feature loss function.
        self.vgg_model = keras.models.Model(inputs=self.vgg.input, outputs=self.vgg.get_layer("block5_conv4").output)

        # ## 3.3. Optimizers

        # Define a learning rate decay schedule.
        self.lr = learning_rate
        # * 0.95 ** ((10 * 1200) // 100000)

        gen_schedule = keras.optimizers.schedules.ExponentialDecay(
            self.lr,
            decay_steps=100000,
            decay_rate=0.95, # 95%
            staircase=True
        )

        disc_schedule = keras.optimizers.schedules.ExponentialDecay(
            self.lr * 5,  # TTUR - Two Time Scale Updates
            decay_steps=100000,
            decay_rate=0.95, # 95%
            staircase=True
        )

        self.gen_optimizer = keras.optimizers.Adam(learning_rate=gen_schedule)
        self.disc_optimizer = keras.optimizers.Adam(learning_rate=disc_schedule)

    @tf.function
    def _feature_loss(self, hr, sr):
        """
        Returns Mean Square Error of VGG19 feature extracted original image (y) and VGG19 feature extracted generated image (y_hat).
        Args:
            hr: A tf tensor of original image (y)
            sr: A tf tensor of generated image (y_hat)
        Returns:
            mse: Mean Square Error.
        """
        sr = keras.applications.vgg19.preprocess_input(((sr + 1.0) * 255) / 2.0)
        hr = keras.applications.vgg19.preprocess_input(((hr + 1.0) * 255) / 2.0)
        sr_features = self.vgg_model(sr) / 12.75
        hr_features = self.vgg_model(hr) / 12.75
        mse = keras.losses.MeanSquaredError()(hr_features, sr_features)
        return mse

    @tf.function
    def _pretrain_step(self, x, y):
        """
        Single step of generator pre-training.
        Args:
            gen_model: A compiled generator model.
            x: The low resolution image tensor.
            y: The high resolution image tensor.
        """
        with tf.GradientTape() as tape:
            fake_hr = self.gen_model(x)
            loss_mse = keras.losses.MeanSquaredError()(y, fake_hr)

        grads = tape.gradient(loss_mse, self.gen_model.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads, self.gen_model.trainable_variables))

        return loss_mse


    def pretrain_generator(self, dataset, writer, log_iter=200):
        """Function that pretrains the generator slightly, to avoid local minima.
        Args:
            gen_model: A compiled generator model.
            dataset: A tf dataset object of low and high res images to pretrain over.
            writer: A summary writer object.
        Returns:
            None
        """

        with writer.as_default():
            for _ in range(1):
                for x, y in dataset:
                    loss = self._pretrain_step(x, y)
                    if self.pretrain_iteration % log_iter == 0:
                        print(f'Pretrain Step: {self.pretrain_iteration}, Pretrain MSE Loss: {loss}')
                        tf.summary.scalar('MSE Loss', loss, step=tf.cast(self.pretrain_iteration, tf.int64))
                        writer.flush()
                    self.pretrain_iteration += 1

    @tf.function
    def _train_step(self, x, y):
        """Single train step function for the SRGAN.
        Args:
            gen_model: A compiled generator model.
            disc_model: A compiled discriminator model.
            x: The low resolution input image.
            y: The desired high resolution output image.
        Returns:
            disc_loss: The mean loss of the discriminator.
            adv_loss: The Binary Crossentropy loss between real label and predicted label.
            cont_loss: The Mean Square Error of VGG19 feature extracted original image (y) and VGG19 feature extractedgenerated image (y_hat).
            mse_loss: The Mean Square Error of original image (y) and generated image (y_hat).
        """
        # Label smoothing for better gradient flow
        valid = tf.ones((x.shape[0],) + self.disc_patch)
        fake = tf.zeros((x.shape[0],) + self.disc_patch)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # From low res. image generate high res. version
            fake_hr = self.gen_model(x)

            # Train the discriminators (original images = real / generated = Fake)
            valid_prediction = self.disc_model(y)
            fake_prediction = self.disc_model(fake_hr)

            # Generator loss
            feat_loss = self._feature_loss(y, fake_hr)

            # not helping because it makes adversial loss increase and discriminator loss decrease
            # cont_loss = content_loss(y, fake_hr)

            # Adversarial Loss need to be decreased. Smallen the number to make it decrease faster
            adv_loss = 1e-3 * keras.losses.BinaryCrossentropy()(valid, fake_prediction)
            mse_loss = 1e-1 * keras.losses.MeanSquaredError()(y, fake_hr)
            perceptual_loss = feat_loss + adv_loss + mse_loss

            # Discriminator loss
            valid_loss = keras.losses.BinaryCrossentropy()(valid, valid_prediction)
            fake_loss = keras.losses.BinaryCrossentropy()(fake, fake_prediction)
            disc_loss = tf.add(valid_loss, fake_loss)

            
        # Backprop on Generator
        gen_grads = gen_tape.gradient(perceptual_loss, self.gen_model.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.gen_model.trainable_variables))

        # Backprop on Discriminator
        disc_grads = disc_tape.gradient(disc_loss, self.disc_model.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_grads, self.disc_model.trainable_variables))
        
        return disc_loss, adv_loss, feat_loss, mse_loss

    def train(self, dataset, writer, log_iter=200):
        """
        Function that defines a single training step for the SR-GAN.
        Args:
            gen_model: A compiled generator model.
            disc_model: A compiled discriminator model.
            dataset: A tf data object that contains low and high res images.
            log_iter: Number of iterations after which to add logs in 
                    tensorboard.
            writer: Summary writer
        """

        with writer.as_default():
            # Iterate over dataset
            for x, y in dataset:
                disc_loss, adv_loss, feat_loss, mse_loss = self._train_step(x, y)
    #             print(self.pretrain_iteration)
                # Log tensorboard summaries if log iteration is reached.
                if self.pretrain_iteration % log_iter == 0:
                    print(f'Train Step: {self.pretrain_iteration}, Adversarial Loss: {adv_loss}, Feature Loss: {feat_loss}, MSE Loss: {mse_loss}, Discriminator Loss: {disc_loss}')
                    
                    tf.summary.scalar('Adversarial Loss', adv_loss, step=self.pretrain_iteration)
                    tf.summary.scalar('Feature Loss', feat_loss, step=self.pretrain_iteration)
                    # tf.summary.scalar('Content Loss', cont_loss, step=self.pretrain_iteration)
                    tf.summary.scalar('MSE Loss', mse_loss, step=self.pretrain_iteration)
                    tf.summary.scalar('Discriminator Loss', disc_loss, step=self.pretrain_iteration)

                    if self.pretrain_iteration % (log_iter*10) == 0:
                        tf.summary.image('Low Res', tf.cast(255 * x, tf.uint8), step=self.pretrain_iteration)
                        tf.summary.image('High Res', tf.cast(255 * (y + 1.0) / 2.0, tf.uint8), step=self.pretrain_iteration)
                        tf.summary.image('Generated', tf.cast(255 * (self.gen_model.predict(x) + 1.0) / 2.0, tf.uint8), step=self.pretrain_iteration)

                    self.gen_model.save(self.gen_model_save_path)
                    self.disc_model.save(self.disc_model_save_path)
                    writer.flush()
                self.pretrain_iteration += 1