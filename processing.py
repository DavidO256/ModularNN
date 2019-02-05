from multiprocessing import Process


class LayerWorker(Process):

    def __init__(self):
        super(LayerWorker, self).__init__()
