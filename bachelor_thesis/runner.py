import bachelor_thesis.model
import skeltorch
import torch.nn.functional as F
import torch.optim


class BachelorThesisRunner(skeltorch.Runner):

    def init_model(self, device):
        self.model = bachelor_thesis.model.BachelorThesisToyModel(
            self.experiment.data.words_embeddings,
            self.experiment.configuration.get('model', 'embedding_size'),
            self.experiment.configuration.get('model', 'lstm_hidden_size'),
            self.experiment.configuration.get('model', 'dropout_prob')
        ).to(device)

    def init_optimizer(self, device):
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train_step(self, it_data, device):
        (news_indexes, news_lengths), prediction_target = it_data
        prediction = self.model(news_indexes.to(device), news_lengths.to(device))
        return F.cross_entropy(prediction, prediction_target.to(device))

    def test(self, epoch, device):
        raise NotImplemented
