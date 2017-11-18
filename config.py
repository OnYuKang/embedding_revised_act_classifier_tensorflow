import argparse
import configparser
import os
import sys
from collections import namedtuple
import utils


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


def str2list(s):
    eval_s = eval(s)
    return eval_s if type(eval_s) == list else []


def get_save_root(config):
    save_root = "%s_train%s_epoch%s_lr%s_unk%s" % (
        config.embed_name.split(".")[0],
        config.train_embed,
        config.num_epochs,
        config.learning_rate,
        config.use_unk
    )
    return save_root


def _get_config():
    config, _ = parser.parse_known_args()
    return config


def get_config():
    return config


def confirm(config):
    try:
        d = vars(config)
    except:
        d = config._asdict()
    
    print("========== RUN CONFIGURATION ==========")
    for key, value in sorted(d.items()):
        print("%s: %s" % (key, value))
    print("=" * 30)
    
    confirm = input("* Run program with current configuration? (y/n) ")

    if confirm.strip().lower() != 'y':
        sys.exit()


arg_lists = []
parser = argparse.ArgumentParser()


# Embedding
embed_args = add_argument_group("Embedding")
embed_args.add_argument("--embed_dir", type=str, default="glove",
                       help="directory where embedding file is located")
embed_args.add_argument("--embed_name", type=str,
                       help="name of the embedding file",
                        default="glove.txt")
embed_args.add_argument("--norm_embed", type=str2bool, default=False,
                       help="boolean flag whether to normalize embedding")
embed_args.add_argument("--norm", type=float, default=1.0,
                       help="normalization factor")
embed_args.add_argument("--eps", type=float, default=1e-7, help="epsilon value")
embed_args.add_argument("--train_embed", type=str2bool, default=False,
                       help="boolean flag whether to train embedding")
embed_args.add_argument("--pad_word", type=str, default="<PAD>",
                      help="special token for padding word")
embed_args.add_argument("--use_unk", type=str2bool, default=False,
                       help="boolean flag whether to use unknown word")
embed_args.add_argument("--unk_word", type=str, default="<UNK>",
                      help="special token for unknown word")
embed_args.add_argument("--max_vocab_size", type=int, default="400000",
                        help="max size of vocabulary")


# Setting
setting_args = add_argument_group("Ontology")
setting_args.add_argument("--ontology_dir", type=str,
                      default="data",
                      help="directory where ontology data file is located")
setting_args.add_argument("--ontology_name", type=str,
                      help="name of the ontology data file",
                       default="ontology.json")
setting_args.add_argument("--csv_setting_dir", type=str, default="data",
                          help="directory where csv setting data file is located")
setting_args.add_argument("--csv_setting_name", type=str, default="csv_setting.json",
                          help="name of the csv setting data file")


# Data
data_args = add_argument_group("Data")
data_args.add_argument("--data_dir", type=str,
                       default="data",
                      help="directory where data files are located")
data_args.add_argument("--train_name", type=str,
                       help="name of the train data file",
                       default="train.csv")
data_args.add_argument("--dev_name", type=str,
                       help="name of the dev data file",
                       default="dev.csv")
data_args.add_argument("--test_name", type=str,
                       help="name of the test data file",
                       default="test.csv")
data_args.add_argument("--has_hyps", type=str2bool, default=True,
                      help="boolean flag whether test data has asr hypotheses")
data_args.add_argument("--num_hyps", type=int, default=10,
                      help="number of hypotheses in test data")
data_args.add_argument("--max_utterance_len", type=eval, default=26,
                      help="Maximum utterance length")


# Model
model_args = add_argument_group("Model")
model_args.add_argument("--num_filters", type=int, default=300, help="number of filters")
model_args.add_argument("--filters_sizes", type=str2list, default="[1, 2, 3]", 
                        help="sizes of filters")
model_args.add_argument("--fnn_activation", type=str, default="sigmoid",
                        choices=["sigmoid", "relu"],
                       help="activation function of FNN networks")
model_args.add_argument("--fnn_dims", type=str2list, default=[100, 2],
                        help="dimensions of layers of FNN")
model_args.add_argument("--dropout_rate", type=float, default=0.5, help="dropout rate")
model_args.add_argument("--grad_clip", type=str2list, default="[-2.0, 2.0]",
                       help="gradient clipping min and max values: [min_grad, max_grad]")
model_args.add_argument("--lmbda", type=float, default=0.55,
                       help="lambda factor")
model_args.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                        help="optimizer")
model_args.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")


# Train
train_args = add_argument_group("Train")
train_args.add_argument("--num_epochs", type=int, default=100, help="number of epochs to run")
train_args.add_argument("--batch_size", type=int, default=256, help="size of a batch")
train_args.add_argument("--pos_rate", type=eval, default=1/8, 
                        help="ratio of positive examples")


# save
save_args = add_argument_group("Save")
save_args.add_argument("--save_step", type=int, default=100, help="steps to save model after")
save_args.add_argument("--save_name", type=str, default='model.ckpt', help="prefix of the saved models")


# Log
log_args = add_argument_group("Log")
log_args.add_argument("--log_dir", type=str, default='logs', help="directory to save log files")
log_args.add_argument("--use_log_format", type=str2bool, default=True,
                      help="boolean flag whether to use '[LEVEL|FILE]TIME > ' format for logs")
log_args.add_argument("--log_to_file", type=str2bool, default=True,
                      help="boolean flag whether to write logs to file")


# Misc
misc_args = add_argument_group("Misc")
misc_args.add_argument("--debug", type=str2bool, default=False,
                       help="boolean flag whether to run in debug mode")
misc_args.add_argument("--gpu_num", type=int, default=0, help="number of GPU to assign")
misc_args.add_argument("-c", "--config_file", help="configuration file", metavar='FILE')
misc_args.add_argument('--slots', nargs='+', help="slots to run for")


config = _get_config()
if config.config_file:
    file_name = config.config_file
    conf = configparser.ConfigParser()
    conf.read(config.config_file)
    
    config_from_file = {}
    for section in conf.sections():
        config_from_file.update(dict([(key, eval(item)) for key, item in conf.items(section)]))
    config_from_file['save_root'] = None
    config_from_file['start_time'] = utils.get_current_time_str()
    config_from_file = namedtuple("Config", config_from_file.keys())(*config_from_file.values())
    
    save_root = get_save_root(config_from_file)
    config_from_file = config_from_file._replace(save_root=save_root)
    confirm(config_from_file)
    
    logger = utils.get_named_logger(config_from_file, 'config')
    logger.info("========== RUN CONFIGURATION: %s ==========" % file_name)
    for key, value in sorted(config_from_file._asdict().items()):
        logger.info("%s: %s" % (key, value))
    logger.info("=" * 30)
    config = config_from_file
    
else:
    save_root = get_save_root(config)
    setattr(config, 'save_root', save_root)
    setattr(config, 'start_time', utils.get_current_time_str())
    confirm(config)
    
    logger = utils.get_named_logger(config, 'config')
    logger.info("========== RUN CONFIGURATION: command line ==========")
    for key, value in sorted(vars(config).items()):
        logger.info("%s: %s" % (key, value))
    logger.info("=" * 30)

if config.gpu_num is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % config.gpu_num
