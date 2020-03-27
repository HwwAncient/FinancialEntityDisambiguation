import matplotlib.pyplot as plt


class DataPainter:
    def __init__(self):

        self.times = 1
        self.epoch = 0
        self.train_loss_x = []
        self.train_loss_y = []
        self.test_loss_y = []
        self.accuracy_y = []
        self.epoch_x = []
        self.xticks = [x for x in range(1, 11)]
        self.xticks_str = [str(int(x)) for x in self.xticks]
        self.axis = [0, 10, 0, 150]

        plt.ion()

        plt.figure(1, figsize=(7.5, 4.5))
        plt.subtitle = "Train Loss"
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.axis(self.axis)
        plt.xticks(self.xticks, self.xticks_str)

        plt.figure(2, figsize=(7.5, 4.5))
        plt.subtitle = "Accuracy"
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.axis([0, 100, 50, 100])

        plt.figure(3, figsize=(7.5, 4.5))
        plt.subtitle = "Test Loss"
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.axis([0, 100, 0, 900])

        plt.show()

    def update(self):
        plt.figure(1)
        self.times = (self.epoch+1)/10 + 1
        self.xticks = [self.times*x for x in range(1, 11)]
        self.xticks_str = [str(int(x)) for x in self.xticks]
        self.axis[1] = 10*self.times
        plt.axis(self.axis)
        plt.xticks(self.xticks, self.xticks_str)
        plt.show()

    def add_train_loss(self, x, y, epoch):
        plt.figure(1)
        self.epoch = epoch
        self.train_loss_x.append(x)
        self.train_loss_y.append(y)

        plt.cla()

        plt.subtitle = "Train Loss"
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.axis(self.axis)
        plt.xticks(self.xticks, self.xticks_str)

        plt.plot(self.train_loss_x, self.train_loss_y, color="blue", linewidth=1.0, linestyle="-")

        plt.pause(0.1)

        if (self.epoch-9) % 10 == 0:
            self.update()

    def add_test(self, accuracy, test_loss, epoch):

        self.test_loss_y.append(test_loss)
        self.accuracy_y.append(accuracy)
        self.epoch_x.append(epoch)

        plt.figure(2)
        plt.cla()
        plt.subtitle = "Accuracy"
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.axis([0, 100, 50, 100])
        plt.plot(self.epoch_x, self.accuracy_y)

        plt.figure(3)
        plt.cla()
        plt.subtitle = "Test Loss"
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.axis([0, 100, 0, 900])
        plt.plot(self.epoch_x, self.test_loss_y)

        plt.pause(0.1)