from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW

from config import Config

# Set seed
pl.trainer.seed.seed_everything(seed=Config.seed)

class IntentDataset(Dataset):
    def __init__(self, queries, targets, tokenizer, max_length):
        """
        Performs initialization of tokenizer
        :param queries: User queries
        :param targets: labels
        :param tokenizer: bert tokenizer
        :param max_length: maximum length of the news text
        """
        self.queries = queries
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        :return: returns the number of datapoints in the dataframe
        """
        return len(self.queries)

    def __getitem__(self, item):
        """
        Returns the review text and the targets of the specified item
        :param item: Index of sample review
        :return: Returns the dictionary of review text, input ids, attention mask, targets
        """
        review = str(self.queries[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


class TransformerDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super(TransformerDataModule, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.MAX_LEN = 100
        self.encoding = None
        self.tokenizer = None
        self.args = kwargs

    def to_label(self, intent):
        """
        Convert intent to integer  based on index in among all indexes.
        """
        return self.classes.index(intent)

    def prepare_data(self):
        """
        Implementation of abstract class
        """

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data
        :param stage: Stage - training or testing
        """
        # reading  the input
        train = pd.read_csv(Config.DATASET_PATH/"train.csv")
        valid = pd.read_csv(Config.DATASET_PATH/"valid.csv")
        test = pd.read_csv(Config.DATASET_PATH/"test.csv")
        dfs = [train,valid,test]

        self.classes = list(set(train.intent.unique().tolist()+valid.intent.unique().tolist()+test.intent.unique().tolist()))

        train,valid,test = [df.intent.apply(self.to_label) for df in dfs]

        self.df_train = train
        self.df_test = test
        self.df_val = valid

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Returns the review text and the targets of the specified item
        :param parent_parser: Application specific parser
        :return: Returns the augmented arugument parser
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for training (default: 16)",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=3,
            metavar="N",
            help="number of workers (default: 0)",
        )
        return parser

    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        """
        Generic data loader function
        :param df: Input dataframe
        :param tokenizer: bert tokenizer
        :param max_len: Max length of the news datapoint
        :param batch_size: Batch size for training
        :return: Returns the constructed dataloader
        """
        ds = IntentDataset(
            queries=df.description.to_numpy(),
            targets=df.label.to_numpy(),
            tokenizer=tokenizer,
            max_length=max_len,
        )

        return DataLoader(
            ds, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"]
        )

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        self.train_data_loader = self.create_data_loader(
            self.df_train, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.train_data_loader

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        self.val_data_loader = self.create_data_loader(
            self.df_val, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.val_data_loader

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        self.test_data_loader = self.create_data_loader(
            self.df_test, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.test_data_loader