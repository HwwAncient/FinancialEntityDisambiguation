
import argparse
import torch.optim as optim

from source.utils.log import *
from source.inputters.lstm_dataset import LSTMDataset
from source.models.NeuPL import *
from source.inputters.corpus import *
from source.utils.misc import str2bool


def model_config():
    """
    model_config
    """
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_dir", type=str, default="./data/")
    data_arg.add_argument("--data_prefix", type=str, default="demo")
    data_arg.add_argument("--save_dir", type=str, default="./models/")
    data_arg.add_argument("--dataset", type=LSTMDataset, default=LSTMDataset)

    # Network
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--embed_size", type=int, default=300)
    net_arg.add_argument("--hidden_size", type=int, default=100)
    net_arg.add_argument("--bidirectional", type=str2bool, default=True)
    net_arg.add_argument("--num_layers", type=int, default=2)
    net_arg.add_argument("--mlp_hidden_size", type=int, default=100)
    net_arg.add_argument("--attn", type=str, default='mlp',
                         choices=['none', 'mlp', 'dot', 'general'])
    net_arg.add_argument("--encoder_model", type=str, default='GRU')

    # Training / Testing
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--optimizer", type=str, default="SGD")
    train_arg.add_argument("--optimizer_weight_decay", type=float, default=0.1)
    train_arg.add_argument("--optimizer_momentum", type=float, default=0.6)
    train_arg.add_argument("--lr", type=float, default=0.001)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--dropout", type=float, default=0)
    train_arg.add_argument("--num_epochs", type=int, default=40)

    # Geneation
    gen_arg = parser.add_argument_group("Generation")
    gen_arg.add_argument("--max_dec_len", type=int, default=30)
    gen_arg.add_argument("--ignore_unk", type=str2bool, default=True)
    gen_arg.add_argument("--length_average", type=str2bool, default=True)
    gen_arg.add_argument("--gen_file", type=str, default="./test.result")
    gen_arg.add_argument("--gold_score_file", type=str, default="./gold.scores")

    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument("--gpu", type=int, default=0)
    misc_arg.add_argument("--batch_size", type=int, default=256)
    misc_arg.add_argument("--ckpt", type=str, default="models/best.32.model")
    misc_arg.add_argument("--check", action="store_true")
    misc_arg.add_argument("--test", action="store_true")
    misc_arg.add_argument("--log_path", default="output/info.log")


    config = parser.parse_args()

    return config


def main():
    """
    main
    """
    config = model_config()

    # logger
    logger = getLogger(config.log_path)

    if config.check:
        config.save_dir = "./tmp/"

    config.use_gpu = torch.cuda.is_available() and config.gpu >= 0
    if config.use_gpu:
        device = config.gpu
        torch.cuda.set_device(device)
        logger.info("model using gpu:{}".format(device))
    else:
        logger.info("model using cpu")


    # Data definition
    corpus = Corpus(data_dir=config.data_dir, data_prefix=config.data_prefix, dataset=config.dataset, logger=logger)

    corpus.load()

    # Embedding
    embed = corpus.embeds


    # if config.test and config.ckpt:
    #     corpus.reload(data_type='test')
    train_iter = corpus.create_batches(config.batch_size, "train", shuffle=True)
    test_iter = corpus.create_batches(config.batch_size, "test", shuffle=False)

    # Model definition
    model = NeuPL(embedding_dim=config.embed_size,
                  hidden_size=config.hidden_size,
                  mlp_hidden_size=config.mlp_hidden_size,
                  mlp_layer_num=config.num_layers,
                  encoder_model=config.encoder_model,
                  dropout=config.dropout)

    model.cuda()
    model_name = model.__class__.__name__

    # Loss Function
    bce = nn.BCELoss(reduction='sum')

    # Optimiser
    optimizer = getattr(torch.optim, config.optimizer)(
        model.parameters(), lr=config.lr, weight_decay=0.01, momentum=0.6)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=0.6, patience=3,
        verbose=False, threshold=0.0001,
        threshold_mode='rel', cooldown=0,
        min_lr=0.0001, eps=5e-04)

    # Train
    logger.info("Train settings: lr-{0} batch_size-{1} epochs-{2} optim-{3} dropout-{4}"
                .format(config.lr, config.batch_size, config.num_epochs, config.optimizer, config.dropout))
    logger.info("Preparation finished, start to train")
    for epoch in range(config.num_epochs):
        logger.info("epoch-{} start".format(epoch+1))
        running_loss = 0
        model.train()
        for i, batch in enumerate(train_iter):
            left_context = embed(batch['input_left']).cuda()
            right_context = embed(batch['input_right']).cuda()
            entity = embed(batch['input_mention']).cuda()
            left_context_lengths = batch['left_length'].cuda()
            right_context_lengths = batch['right_length'].cuda()
            entity_description = embed(batch['input_describe']).cuda()
            entity_description_lengths = batch['describe_length'].cuda()
            target = batch['target'].cuda()


            try:
                out = model(left_context,
                            left_context_lengths,
                            right_context,
                            right_context_lengths,
                            entity_description, entity_description_lengths, entity)
            except RuntimeError:
                logger.error("ModelRunning unknown error")
                model.save(config.ckpt)
                continue

            optimizer.zero_grad()

            assert out.size() == target.size()

            loss = bce(out, target)

            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()

            if i % 10 == 9:
                logger.info('[epoch-{0} batch-{1}] train-loss: {2}'.format(epoch + 1, i + 1, "%.4f" % (running_loss / 10)))
                running_loss = 0

        test_loss = 0
        right = 0
        test_length = 0

        model.eval()
        for j, test_batch in enumerate(test_iter):
            left_context = embed(test_batch['input_left']).cuda()
            right_context = embed(test_batch['input_right']).cuda()
            entity = embed(test_batch['input_mention']).cuda()
            left_context_lengths = test_batch['left_length'].cuda()
            right_context_lengths = test_batch['right_length'].cuda()
            entity_description = embed(test_batch['input_describe']).cuda()
            entity_description_lengths = test_batch['describe_length'].cuda()
            target = test_batch['target'].cuda()

            out = model(left_context,
                        left_context_lengths,
                        right_context,
                        right_context_lengths,
                        entity_description, entity_description_lengths, entity)

            loss = bce(out, target)
            test_loss += loss.data.item()

            out_list = out.cpu().squeeze(1).squeeze(1).data.numpy().tolist()
            target_list = target.cpu().squeeze(1).squeeze(1).data.numpy().tolist()

            for x, y in zip(out_list, target_list):
                if abs(x-y) < 0.5:
                    right += 1
                test_length += 1
        lr_scheduler.step(test_loss)
        logger.info('[epoch-{0}] test-loss: {1} test-accuracy: {2}'
                    .format(epoch + 1, "%.4f" % test_loss, "%.4f" % (right/test_length)))

    model.save(config.ckpt)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")