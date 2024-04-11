class ModelParams:
    def __init__(self, optimizer, loss_fn, metrics) -> None:
        self.metrics = None
        self.optimizer = None
        self.loss_fn = None
        self.set_params(optimizer, loss_fn, metrics)

    def set_params(self, optimizer, loss_fn, metrics):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics


class ResNet50(ModelParams):
    def __init__(self, optimizer='adam', loss_fn='categorical_crossentropy', metrics=['accuracy']) -> None:
        super().__init__(optimizer, loss_fn, metrics)