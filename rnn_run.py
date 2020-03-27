
import argparse
import torch.optim as optim
import matplotlib.pyplot as plt

from source.utils.log import *
from source.inputters.lstm_dataset import LSTMDataset
from source.models.NeuPL import *
from source.inputters.corpus import *
from source.utils.misc import str2bool
from source.utils.draw import *

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
    net_arg.add_argument("--entity_hidden_size", type=int, default=400)
    net_arg.add_argument("--context_hidden_size", type=int, default=400)
    net_arg.add_argument("--bidirectional", type=str2bool, default=True)
    net_arg.add_argument("--num_layers", type=int, default=2)
    net_arg.add_argument("--mlp_hidden_size", type=int, default=400)
    net_arg.add_argument("--attn", type=str, default='mlp',
                         choices=['none', 'mlp', 'dot', 'general'])
    net_arg.add_argument("--encoder_model", type=str, default='GRU')
    net_arg.add_argument("--encoder_layers", type=int, default=1)

    # Training / Testing
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--optimizer", type=str, default="SGD")
    train_arg.add_argument("--optimizer_weight_decay", type=float, default=0.75)
    train_arg.add_argument("--optimizer_momentum", type=float, default=0.8)
    train_arg.add_argument("--lr", type=float, default=0.001)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--dropout", type=float, default=0)
    train_arg.add_argument("--num_epochs", type=int, default=150)

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
    misc_arg.add_argument("--ckpt", type=str, default="models/best.128.1.model")
    misc_arg.add_argument("--check", action="store_true")
    misc_arg.add_argument("--test", action="store_true")
    misc_arg.add_argument("--log_path", default="output/info.log")


    config = parser.parse_args()

    return config

def LINEST(points: list):
    n = len(points)
    xs = [i for i in range(1, n+1)]
    y = sum(points) / n
    x = sum(xs) / n
    sum_xy = 0
    sum_x2 = 0
    for i in xs:
        sum_xy += i*points[i-1]
        sum_x2 += i**2

    out = (sum_xy-n*x*y)/(sum_x2-n*(x**2))

    return out

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
    train_iter = corpus.create_batches(config.batch_size, "train.large", shuffle=True)
    test_iter = corpus.create_batches(config.batch_size, "test.large", shuffle=False)

    # Model definition
    model = NeuPL(embedding_dim=config.embed_size,
                  entity_hidden_size=config.entity_hidden_size,
                  context_hidden_size=config.context_hidden_size,
                  mlp_hidden_size=config.mlp_hidden_size,
                  encoder_model=config.encoder_model,
                  dropout=config.dropout,
                  encoder_layers=config.encoder_layers)

    model.cuda()
    model_name = model.__class__.__name__

    # Loss Function
    bce = nn.BCELoss(reduction='sum')

    # Optimiser
    # optimizer = getattr(torch.optim, config.optimizer)(
    #     model.parameters(), lr=config.lr,
    #     weight_decay=config.optimizer_weight_decay,
    #     momentum=config.optimizer_momentum)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.optimizer_weight_decay)

    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min',
    #     factor=0.6, patience=3,
    #     verbose=False, threshold=0.0001,
    #     threshold_mode='rel', cooldown=0,
    #     min_lr=0.0001, eps=5e-04)

    lr_scheduler_step_by = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)

    lr_scheduler_step = optim.lr_scheduler.MultiStepLR(optimizer, [15, 50, 120], gamma=0.3)

    painter = DataPainter()

    accuracy = 0

    train_loss = []

    # Train
    logger.info("Train settings: lr-{0} batch_size-{1} epochs-{2} dropout-{3}"
                .format(config.lr, config.batch_size, config.num_epochs, config.dropout))
    logger.info("Optimizer settings: optim-{0} weight_decay-{1} momentum-{2}"
                .format(config.optimizer, config.optimizer_weight_decay, config.optimizer_momentum))
    logger.info("Model settings: entity_hs-{0} context_hs-{1} mlp_hs-{2} encoder-{3} encoder_layer-{4}"
                .format(config.entity_hidden_size, config.context_hidden_size, config.mlp_hidden_size,
                        config.encoder_model, config.encoder_layers))

    logger.info("Preparation finished, start to train")
    for epoch in range(config.num_epochs):
        logger.info("epoch-{} start".format(epoch+1))
        running_loss = 0
        step = 0
        model.train()
        train_size = corpus.data_size(data_type='train.large')
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
                step -= 1
                train_loss.append(running_loss)

                if len(train_loss) > 25:
                    lin = LINEST(train_loss[-25:])
                elif len(train_loss) == 1:
                    lin = 0.0
                else:
                    lin = LINEST(train_loss)

                if lin > 0.001:
                    if step <= 0:
                        lr_scheduler_step_by.step()
                        step = 15

                logger.info('[epoch-{0} batch-{1}] train-loss: {2} LINEST: {3}' .
                            format(epoch + 1, i + 1, "%.4f" % (running_loss / 10), "%.4f" % lin))
                painter.add_train_loss((i + 1) / (train_size / config.batch_size) + epoch, running_loss/10, epoch+1)
                running_loss = 0


        lr_scheduler_step.step()
        # lr_scheduler_step_by.step()

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
        # lr_scheduler.step(test_loss)
        logger.info('[epoch-{0}] test-loss: {1} test-accuracy: {2}'
                    .format(epoch + 1, "%.4f" % test_loss, "%.4f" % (right/test_length)))

        if right/test_length > accuracy:
            accuracy = right/test_length
            model.save('models/7/test.128.best.model')
        else:
            model.save('models/7/test.128.model')

        painter.add_test(right*100/test_length, test_loss, epoch+1)

    model.save(config.ckpt)





if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")