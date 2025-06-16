import torch.nn as nn
from src.system_model import SystemModel
from src.criterions import RMSPELoss


class ParentModel(nn.Module):
    def __init__(self, system_model: SystemModel, training_loss_type):

        super(ParentModel, self).__init__()
        self.system_model = system_model
        self._set_criterion(training_loss_type.lower())

    def get_model_name(self):
        return f"{self._get_name()}_{self.get_model_params()}"

    def get_model_params(self):
        return None

    def training_step(self, batch):
        raise NotImplementedError

    def validation_step(self, batch):
        raise NotImplementedError

    def test_step(self, batch):
        raise NotImplementedError

    def _set_criterion(self, training_loss_type):
        if training_loss_type == "rmspe":
            self.criterion = RMSPELoss()
        else:
            raise NotImplementedError(f"The loss: {training_loss_type} isn't supported")

    def get_model_file_name(self):
        M = str(self.system_model.params.M).replace(' ', '-')
        snr = str(self.system_model.params.snr).replace(' ', '-')
        return f"{self.get_model_name()}_" + \
            f"N={self.N}_" + \
            f"M={M}_" + \
            f"T={self.system_model.params.T}_" + \
            f"{self.system_model.params.signal_type}_" + \
            f"SNR={snr}_" + \
            f"{self.system_model.params.field_type}_field_" + \
            f"{self.system_model.params.signal_nature}_" + \
            f"eta={self.system_model.params.eta}_" + \
            f"sv_var={self.system_model.params.sv_noise_var}"


if __name__ == "__main__":
    pass
