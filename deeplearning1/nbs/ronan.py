import PIL
import tqdm
import pandas as pd
import json
import chainer
import os
import re
import numpy as np
import contextlib
import matplotlib
import six
import matplotlib.pyplot as plt
import logging
# plt.style.use('ggplot')

log = logging.getLogger('ronan')
logging.basicConfig(level=logging.DEBUG)

chainer.cuda.get_device_from_id(0).use()

@contextlib.contextmanager
def maybe_backprop(backprop):
    if backprop:
        yield
    else:
        with chainer.no_backprop_mode():
            yield


def call_finetune(x, net, backprop_from_layer, target_layer):
    """Same idea as ResNet50Layers.__call__(), but allows starting
    backprop from a given layer. Also, a single output layer is
    given and its activation is returned directly.

    Example, to finetune layer res5 and everything up until pool5:
      ```pool5_output = call_finetune(resnet50, 'res5', 'pool5')```

    """
    backprop = False
    h = x
    for key, functions in six.iteritems(net.functions):
        if key == backprop_from_layer:
            backprop = True
        with maybe_backprop(backprop):
            for function in functions:
                h = function(h)
        if key == target_layer:
            return h



class ResNet50(chainer.Chain):
    """Download pretrained resnet50 from
    'https://github.com/KaimingHe/deep-residual-networks' (One Drive
    link:
    https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)
    ) and place it on
    ~/.chainer/dataset/pfnet/chainer/models/ResNet-50-model.caffemodel

    """

    def __init__(self, output_size=2, finetune_from_layer='pool5'):
        super(ResNet50, self).__init__()
        self.finetune_from_layer = finetune_from_layer
        with self.init_scope():
            self.resnet50 = chainer.links.ResNet50Layers()
            self.fc_final = chainer.links.Linear(None, output_size)

    def __call__(self, x):
        y = call_finetune(x, self.resnet50, self.finetune_from_layer, 'pool5')
        return self.fc_final(y)

class ResNet50Features(chainer.Chain):
    def __init__(self, output_layer='pool5', finetune_from_layer='_no_finetuning_thanks_'):
        super(ResNet50Features, self).__init__()
        self.finetune_from_layer = finetune_from_layer
        self.output_layer = output_layer
        with self.init_scope():
            self.resnet50 = chainer.links.ResNet50Layers()

    def __call__(self, x):
        return call_finetune(x, self.resnet50,
                             backprop_from_layer=self.finetune_from_layer,
                             target_layer=self.output_layer)

    
class DualNet(chainer.Chain):
    def __init__(self, base, output_size=2):
        super(DualNet, self).__init__()
        with self.init_scope():
            self.base = base
            self.fc_final = chainer.links.Linear(None, output_size)

    def __call__(self, x):
        assert x.shape[1] == 2
        # (we receive (batch, 2, channels, height, width))
        # x = x.transpose(1, 0, 2, 3)
        # i1, i2 = x
        i1 = x[:, 0, :, :]
        i2 = x[:, 1, :, :]
        h = chainer.functions.concat((self.base(i1),
                                      self.base(i2)))
        return self.fc_final(h)
    
def Resize(size):
    def transform(image):
        label = None
        if len(image) == 2:
            image, label = image
        if image.shape[0] in [3, 1]:
            image = image.transpose(1, 2, 0)
        image = PIL.Image.fromarray(image.astype(np.uint8))
        image = image.resize(size)
        if label is None:
            return np.asarray(image).transpose(2, 0, 1).astype(np.float32)
        else:
            return (np.asarray(image).transpose(2, 0, 1).astype(np.float32),
                    label)

    return transform


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def image_dataset(path_components, resize=(224, 224), random_rotate=False):
    path = os.path.join(*path_components)
    images = os.listdir(path)
    dataset = chainer.datasets.ImageDataset(images, root=path)
    if resize is not None:
        dataset = chainer.datasets.TransformDataset(dataset, Resize(resize))
    if random_rotate:
        import chainercv
        dataset = chainer.datasets.TransformDataset(dataset, chainercv.transforms.random_rotate)
    dataset.files = images
    dataset.path = path
    return dataset


def load_cats_datasets(
        data_path='../../data/kaggle/dogs-vs-cats-redux-kernels-edition/',
        proportion_train=0.7,
        seed=42):
    data = data_path

    path_train = os.path.join(data, 'train')
    all_images = os.listdir(path_train)
    all_labels = np.asarray(['dog' in x for x in all_images], dtype=np.int32)
    full_dataset = chainer.datasets.LabeledImageDataset(
        zip(all_images, all_labels), root=path_train)
    full_dataset = chainer.datasets.TransformDataset(full_dataset,
                                                     Resize((224, 224)))
    train_dataset, valid_dataset = chainer.datasets.split_dataset_random(
        full_dataset,
        first_size=int(len(full_dataset) * proportion_train),
        seed=seed)
    test_dataset = image_dataset([data, 'test'])

    return AttrDict(
        train=train_dataset,
        validation=valid_dataset,
        test=test_dataset,
        root=data_path,
        id_name = 'id',
        label_names=['cat', 'dog'])


def load_galaxy_datasets(
        data_path='../../../data/kaggle/galaxy-zoo-the-galaxy-challenge/',
        proportion_train=0.7):
    datasets = AttrDict()
    datasets.test = image_dataset([data_path, 'images_test_rev1'])

    labeled_path = 'images_training_rev1'
    labels = pd.read_csv(
        os.path.join(data_path, 'training_solutions_rev1.csv'))
    images = ["{}/{}.jpg".format(labeled_path, galaxy) for galaxy in labels.GalaxyID]
    labeled_dataset = chainer.datasets.LabeledImageDataset(
        zip(images, labels[range(1, len(labels.columns))].values),
        root=data_path, label_dtype=np.float32)
    labeled_dataset = chainer.datasets.TransformDataset(labeled_dataset, Resize((224, 224)))

    datasets.train, datasets.validation = chainer.datasets.split_dataset_random(
        labeled_dataset, first_size=int(len(labeled_dataset) * proportion_train),
        seed=42)

    datasets.string_of_label = str

    datasets.id_name = labels.columns[0]
    datasets.label_names = labels.columns[1:] 
    
    return datasets


def images_side_by_side(i1, i2, output_size=(224, 224)):
    size1 = (output_size[0], int(output_size[1]/2)) #(np.array(output_size)/2).astype(np.int)
    size2 = (output_size[0], output_size[1] - size1[1])
    resizer1 = Resize(size1)
    resizer2 = Resize(size2)
    return np.hstack((resizer1(i1), resizer2(i2)))

def image_pair(i1, i2, size):
    resizer = Resize(size)
    return (resizer(i1), resizer(i2))

class ImagePairsDataset(chainer.dataset.DatasetMixin):
    def __init__(self, pairs, root, size=(224, 224), assemble_images=images_side_by_side):
        self.pairs = pairs
        self.root = root
        self.size = size
        self.files = self.pairs.index
        self.assemble_images = assemble_images
        
    def __len__(self):
        return len(self.pairs.index)

    def _read_image(self, column_name, i):
        return np.asarray(PIL.Image.open(os.path.join(self.root, self.pairs[column_name].iloc[i])),
                          dtype=np.float32)
        
    def get_example(self, i):
        image1 = self._read_image("image1", i)
        image2 = self._read_image("image2", i)
        if 'label' in self.pairs.columns:
            return (self.assemble_images(image1, image2, self.size), np.int32(self.pairs.label.iloc[i]))
        else:
            return self.assemble_images(image1, image2, self.size)

def SomeImagePairsDataset(labels, root, num_pairs, size=(224, 224), assemble_images=images_side_by_side):
    """We want to pick num_pairs/2 with same artist, and num_pairs/2
    completely at random.
    
    """
    # pick num_pairs/2 images at random
    i_random = np.random.choice(len(labels.index), num_pairs).reshape((-1, 2))
    # pick num_pairs/2 images, taking care to have the same artist
    assert 'i' not in labels.columns
    labels["i"] = range(len(labels.index))
    i_same = []
    num_by_artist = int(num_pairs/(2 * len(labels.artist.unique()))) + 1
    # while len(i_same) < num_pairs/2:
    for artist, group in labels.groupby('artist'):
        indices = group.i.sample(2 * num_by_artist, replace=True)
        for i in range(0, len(indices), 2):
            i_same.append((indices.iloc[i], indices.iloc[i+1]))
        # i_same.append(tuple(group.i.sample(2)))
        # i_same.append(group.i.iloc[0] + np.random.choice(len(group.artist), 2, replace=False))
    indexes = np.vstack((i_random, i_same))
    pairs = pd.DataFrame(dict(image1=labels.filename[indexes[:, 0]].values,
                              image2=labels.filename[indexes[:, 1]].values,
                              i=range(indexes.shape[0]),
                              label=(labels.artist[indexes[:, 0]] == labels.artist[indexes[:, 1]])))
    return ImagePairsDataset(pairs, root, size=size, assemble_images=assemble_images)

    
def load_painting_datasets(
        data_path='../../../data/kaggle/painter-by-numbers/',
        proportion_train=0.7, size=1000000,
        assemble_images=images_side_by_side):
    datasets = AttrDict()

    # idea:
    # - labeled: an image dataset with image + painter label
    # - labeled_pairs: from this, a dataset of image pairs + same painter label
    # - train, validation: split labeled_pairs into train+validation
    # - unlabeled: a test dataset of just images
    # - test: test dataset of pairs as described in submission_info.csv

    labels = pd.read_csv(os.path.join(data_path, 'train_info.csv'))
    # log.debug('building labeled dataset')
    # datasets.labeled = chainer.datasets.LabeledImageDataset(zip(labels.filename, labels.artist),
    #                                                         root=os.path.join(data_path, 'train'),
    #                                                         label_dtype=str)
    # log.debug('labeled dataset: len {}'.format(len(datasets.labeled)))
        
    log.debug('building labeled pairs dataset')
    datasets.labeled_pairs = SomeImagePairsDataset(labels, num_pairs=size,
                                                   root=os.path.join(data_path, 'train'),
                                                   assemble_images=assemble_images)

    log.debug('labeled pairs dataset: len {}'.format(len(datasets.labeled_pairs)))
    log.debug('building train, validation split')
    datasets.train, datasets.validation = chainer.datasets.split_dataset_random(
        datasets.labeled_pairs, first_size=int(len(datasets.labeled_pairs) * proportion_train),
        seed=42)

    log.debug('building unlabeled image pair dataset')
    submission_info = pd.read_csv(os.path.join(data_path, 'submission_info.csv'))
    datasets.test = ImagePairsDataset(submission_info, root=os.path.join(data_path, 'test'),
                                      assemble_images=assemble_images)
    
    datasets.id_name = 'index'
    datasets.label_names = ['sameArtist']
    log.debug('done building dataset')

    return datasets
    
def string_of_label(label):
    return ['cat', 'dog'][label]


def plot_dataset_image(image):
    label = None
    if len(image) == 2:
        image, label = image
    image = np.transpose(image, (1, 2, 0)) / 255.
    plt.figure()
    plt.imshow(image, interpolation='nearest')
    plt.axis('off')
    if label is not None:
        plt.title(label)
        # plt.title('{}'.format(string_of_label(label)))

        
class OnlineEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, *args, **kwargs):
        super(OnlineEvaluator, self).__init__(*args, **kwargs)

    @staticmethod
    def _reset_iterator(iterator):
        if hasattr(iterator, 'reset'):
            iterator.reset()
            return iterator
        else:
            return copy.copy(iterator)
        
    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']
        if self.eval_hook:
            self.eval_hook(self)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = self._reset_iterator(iterator)
            batch = next(iterator)

        summary = chainer.reporter.DictSummary()
            
        observation = {}
        with chainer.reporter.report_scope(observation):
            in_arrays = self.converter(batch, self.device)
            with chainer.function.no_backprop_mode():
                if isinstance(in_arrays, tuple):
                    eval_func(*in_arrays)
                elif isinstance(in_arrays, dict):
                    eval_func(**in_arrays)
                else:
                    eval_func(in_arrays)
                    
        summary.add(observation)

        return summary.compute_mean()


def train(predictor,
          data,
          name,
          train_batch_size=128,
          validation_batch_size=512,
          trigger_save_trainer=(1, 'epoch'),
          optimizer=None,
          **kwargs):
    train_iter = chainer.iterators.MultiprocessIterator(
        data.train, batch_size=train_batch_size, shuffle=True)
    validation_iter = chainer.iterators.MultiprocessIterator(
        data.validation,
        batch_size=validation_batch_size,
        repeat=False,
        shuffle=False)

    if 'loss_fun' in kwargs:
        kwargs['lossfun'] = kwargs['loss_fun']
        del kwargs['loss_fun']

    if 'accuracy_fun' in kwargs:
        kwargs['accfun'] = kwargs['accuracy_fun']
        del kwargs['accuracy_fun']
        
    model = chainer.links.Classifier(predictor, **kwargs)
    if 'accfun' in kwargs and kwargs['accfun'] is None:
        log.info('disabling accuracy')
        model.compute_accuracy = False
    model.to_gpu()

    log.debug('setting up optimizer')
    if optimizer is None:
        optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=0)

    # trigger_evaluation = (150, 'iteration')
    trigger_evaluation = (int(validation_batch_size / train_batch_size), 'iteration')
    
    log.debug('creating trainer')
    name = "{}-{}-batch{}".format(predictor.__class__.__name__, name, train_batch_size)
    log.info("full model name: %s", name)
    trainer = chainer.training.Trainer(updater, out=name)
    # trainer.extend(
    #     chainer.training.extensions.Evaluator(
    #         validation_iter, model, device=0),
    #     trigger=trigger_evaluation)
    trainer.extend(
        OnlineEvaluator(
            validation_iter, model, device=0),
        trigger=trigger_evaluation)
    trainer.extend(chainer.training.extensions.LogReport(trigger=(1, 'iteration')))
    trainer.extend(
        chainer.training.extensions.PrintReport([
            'elapsed_time', 'epoch', 'iteration', 'main/accuracy',
            'validation/main/accuracy', 'main/loss', 'validation/main/loss'
        ]),
        trigger=(1, 'iteration'))
    # this saves the complete trainer object
    trainer.extend(
        chainer.training.extensions.snapshot(), trigger=trigger_save_trainer)
    # this saves only the resnet model (hopefuly)
    trainer.extend(
        chainer.training.extensions.snapshot_object(
            model.predictor, 'snapshot_model_{.updater.iteration}'),
        trigger=chainer.training.triggers.MinValueTrigger(
            'validation/main/loss', trigger=trigger_evaluation))

    # todo: if there is test data, produce a test prediction in an extension

    # restart from a saved point
    # chainer.serializers.load_npz('result2/snapshot_iter_1600', trainer)

    log.debug('running training loop')
    trainer.run()


def predict_dog(model, data, batch_size=512):
    test_iter = chainer.iterators.SerialIterator(
        data.test, batch_size=batch_size, repeat=False, shuffle=False)
    probas = []
    for batch in tqdm.tqdm(
            test_iter, total=len(data.test) / test_iter.batch_size):
        # .get() converts the GPU array to a CPU one
        proba_dog = chainer.functions.softmax(
            model(model.xp.asarray(batch))).data[:, 1].get()
        probas.append(proba_dog)
    proba_dog = np.concatenate(probas)
    id_image = [int(os.path.splitext(x)[0]) for x in data.test.files]
    prediction = pd.DataFrame(dict(id=id_image, label=list(proba_dog)))
    prediction.sort_values('id', inplace=True)
    return prediction


def predict(model, data, batch_size=512):
    model.to_gpu()
    test_iter = chainer.iterators.SerialIterator(
        data.test, batch_size=batch_size, repeat=False, shuffle=False)
    outputs = []
    for batch in tqdm.tqdm(
            test_iter, total=len(data.test) / test_iter.batch_size):
        # .get() converts the GPU array to a CPU one
        output = model(model.xp.asarray(batch)).data.get()
        outputs.append(output)
    id_name = data.get('id_name', 'id')
    id_image = pd.DataFrame({id_name: [int(os.path.splitext(x)[0]) for x in data.test.files]})
    output_df = pd.DataFrame(np.vstack(outputs))
    if hasattr(data, 'label_names'):
        output_df.columns = data.label_names
    prediction = pd.concat((id_image, output_df), axis=1)
    prediction.sort_values(id_name, inplace=True)
    return prediction


def plot_prediction(prediction, data, num=10):
    root = data.get('root',
                    '../../data/kaggle/dogs-vs-cats-redux-kernels-edition/')
    for _, row in prediction[:10].iterrows():
        image = PIL.Image.open(
            os.path.join(root, 'test', "{}.jpg".format(int(row.id))))
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.title("{}: dog={}".format(row.id, row.label))


def plot_painting(filename, info, path='../../../data/kaggle/painter-by-numbers/'):
    image = PIL.Image.open(os.path.join(path, filename))
    just_filename = os.path.split(filename)[-1]
    info = info[info.filename == just_filename]
    if len(info.index) > 0:
        title = "[{x.filename}] {x.artist}: {x.title} ({x.date}, {x.style}, {x.genre})".format(x=info)
    else:
        title = filename
        plt.imshow(image)
        plt.axis('off')
        
def read_logs(*result_dirs):
    logs = []
    for result_dir in result_dirs:
        with file('{}/log'.format(result_dir)) as logf:
            logs += json.load(logf)
    log = pd.DataFrame(logs)
    return log


def plot_logs(result_dirs, axes=None, components=['accuracy', 'loss'], name=None,
              x='elapsed_time', alpha_train=0.2, alpha_validation=0.95):
    log = read_logs(*result_dirs)
    if axes is None:
        fig, axes = plt.subplots(len(components), sharex=True, figsize=(14, 8))
    assert len(axes) == len(components)
    if name is None:
        name = os.path.split(result_dirs[0])[-1]
    for component, ax in zip(components, axes):
        train_line = ax.plot(
            log[x],
            log['main/{}'.format(component)],
            #label='{}/train/{}'.format(name, component),
            label='{}/{}'.format(name, component),
            alpha=alpha_train)
        try:
            measurement = 'validation/main/{}'.format(component)
            ax.plot(
                log[x],
                log[measurement].interpolate(),
                # label='{}/validation/{}'.format(name, component),
                # label='{}/{}'.format(name, component),
                label='_nolegend_',
                color=train_line[-1].get_color(),
                lw=2,
                alpha=alpha_validation)
        except KeyError:
            print "measurement not found: {}".format(measurement)
        if x == 'elapsed_time':
            ax.set_xlabel("t (s)")
        else:
            ax.set_xlabel(x)
            # ax.legend()
