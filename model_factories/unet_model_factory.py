import models.unet


class UnetModelFactory:
    def __init__(self, batch_size, optimizer):
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.name = 'unet'

    def get_model(self, input_size=None, pretrained_weights=None):
        return models.unet.get_model(pretrained_weights=pretrained_weights, input_size=input_size,
                                     optimizer=self.optimizer, show_summary=False)

    def get_name(self):
        return self.name + '_' + str(self.batch_size) + '_' + str(self.optimizer.get_config()['learning_rate'])

    def get_batch_size(self):
        return self.batch_size

    def __str__(self):
        return self.get_name()
