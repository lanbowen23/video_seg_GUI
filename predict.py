import os, time, scipy
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def osmn_arg_scope(weight_decay=0.0002):
    """Defines the OSMN arg scope.
    Args:
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer(),
                        biases_regularizer=None,
                        padding='SAME'):
        with slim.arg_scope([slim.avg_pool2d, slim.max_pool2d],
                            padding='SAME') as arg_sc:
            return arg_sc


def visual_modulator(guide_image, model_params, scope='osmn', is_training=False):
    """Defines the visual modulator
    Args:
    guide_image: visual guide image
    model_params: parameters related to model structure
    scope: scope name for the network
    is_training: training or testing
    Returns:
    Tensor of the visual modulation parameters
    """
    mod_early_conv = model_params.mod_early_conv
    n_modulator_param = 512 * 6 + 256 * 3 + mod_early_conv * 384
    with tf.variable_scope(scope, [guide_image]) as sc, slim.arg_scope(osmn_arg_scope()) as arg_sc:
        end_points_collection = sc.name + '_end_points'
        modulator_params = None

        with tf.variable_scope('modulator'):
            # Collect outputs of all intermediate layers.
            with slim.arg_scope([slim.conv2d],
                                padding='SAME',
                                outputs_collections=end_points_collection):
                net = slim.repeat(guide_image, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net_2 = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net_2, [2, 2], scope='pool2')
                net_3 = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net_3, [2, 2], scope='pool3')
                net_4 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net_4, [2, 2], scope='pool4')
                net_5 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net_5, [2, 2], scope='pool5')
                net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
                net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout7')
                modulator_params = slim.conv2d(net, n_modulator_param, [1, 1],
                                               weights_initializer=tf.zeros_initializer(),
                                               biases_initializer=tf.ones_initializer(),
                                               activation_fn=None, normalizer_fn=None, scope='fc8')
                modulator_params = tf.squeeze(modulator_params, [1, 2])
    return modulator_params


def conditional_normalization(inputs, gamma, reuse=None, scope=None):
    with tf.variable_scope(scope, 'ConditionalNorm', [inputs, gamma],
                           reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims

        if inputs_rank is None:
            raise ValueError('Inputs %s has undefined rank.' % inputs.name)
        if inputs_rank != 4:
            raise ValueError('Inputs %s is not a 4D tensor.' % inputs.name)
        params_shape = inputs_shape[-1:]
        gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
        if not params_shape.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (
                inputs.name, params_shape))
        return inputs * gamma


def modulated_conv_block(net, repeat, channels, dilation=1, scope_id=0, visual_mod_id=0,
                         visual_modulation_params=None,
                         spatial_modulation_params=None,
                         visual_modulation=False,
                         spatial_modulation=False):
    spatial_mod_id = 0
    for i in range(repeat):
        net = slim.conv2d(net, channels, [3, 3], rate=dilation,
                          scope='conv{}/conv{}_{}'.format(scope_id, scope_id, i + 1))
        if visual_modulation:
            # NHWC, NC
            vis_params = tf.slice(visual_modulation_params, [0, visual_mod_id], [-1, channels],
                                  name='m_param{}'.format(scope_id))
            net = conditional_normalization(net, vis_params,
                                            scope='conv{}/conv{}_{}'.format(scope_id, scope_id, i + 1))
            visual_mod_id += channels
        if spatial_modulation:
            sp_params = tf.slice(spatial_modulation_params,
                                 [0, 0, 0, spatial_mod_id], [-1, -1, -1, channels],
                                 name='m_sp_param{}'.format(scope_id))
            net = tf.add(net, sp_params)
            spatial_mod_id += channels
    return net, visual_mod_id


def crop_features(feature, out_size):
    """Crop the center of a feature map
    Args:
    feature: Feature map to crop
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    """
    up_size = tf.shape(feature)
    ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
    ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
    slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
    # slice_input = tf.slice(feature, (0, ini_w, ini_w, 0), (-1, out_size[1], out_size[2], -1))  # Caffe cropping way
    return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])


def osmn_deeplab(inputs, model_params, visual_modulator_params=None, scope='osmn', is_training=False):
    """Defines the OSMN with deeplab backbone
    Args:
    inputs: Tensorflow placeholder that contains the input image, visual guide, and spatial guide
    model_params: parameters related to the model structure
    visual_modulator_params: if None it will generate new visual modulation parameters using guide image, otherwise
            it will reuse the current parameters.
    scope: Scope name for the network
    is_training: training or testing
    Returns:
    net: output tensor of the network
    end_points: dictionary with all tensors of the network
    """
    im_size = tf.shape(inputs[2])
    mod_early_conv = model_params.mod_early_conv
    use_visual_modulator = model_params.use_visual_modulator
    use_spatial_modulator = model_params.use_spatial_modulator
    batch_norm_params = {
        'decay': 0.99,
        'scale': True,
        'epsilon': 0.001,
        'updates_collections': None,
        'is_training': not model_params.fix_bn and is_training
    }
    num_mod_layers = [2, 2, 3, 3, 3]
    train_seg = model_params.train_seg
    if use_visual_modulator and visual_modulator_params is None:
        visual_modulator_params = visual_modulator(inputs[0], model_params, scope=scope, is_training=is_training)

    with tf.variable_scope(scope, [inputs]) as sc, slim.arg_scope(osmn_arg_scope()) as arg_sc:
        end_points_collection = sc.name + '_end_points'
        # index to mark the current position of the modulation params
        visual_mod_id = 0
        with tf.variable_scope('modulator_sp'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                padding='SAME',
                                outputs_collections=end_points_collection) as bn_arg_sc:
                if not use_spatial_modulator:
                    conv1_att = None
                    conv2_att = None
                    conv3_att = None
                    conv4_att = None
                    conv5_att = None
                else:
                    ds_mask = slim.avg_pool2d(inputs[1], [2, 2], scope='pool1')
                    if mod_early_conv:
                        conv1_att = slim.conv2d(inputs[1], 64 * num_mod_layers[0], [1, 1], scope='conv1')
                        conv2_att = slim.conv2d(ds_mask, 128 * num_mod_layers[1], [1, 1], scope='conv2')
                    else:
                        conv1_att = None
                        conv2_att = None
                    ds_mask = slim.avg_pool2d(ds_mask, [2, 2], scope='pool2')
                    conv3_att = slim.conv2d(ds_mask, 256 * num_mod_layers[2], [1, 1], scope='conv3')
                    ds_mask = slim.avg_pool2d(ds_mask, [2, 2], scope='pool3')
                    conv4_att = slim.conv2d(ds_mask, 512 * num_mod_layers[3], [1, 1], scope='conv4')
                    conv5_att = slim.conv2d(ds_mask, 512 * num_mod_layers[4], [1, 1], scope='conv5')
        with tf.variable_scope('seg'):
            # Collect outputs of all intermediate layers.
            with slim.arg_scope([slim.conv2d],
                                padding='SAME', trainable=train_seg,
                                outputs_collections=end_points_collection):
                with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                    net_1, visual_mod_id = modulated_conv_block(inputs[2], 2, 64,
                                                                scope_id=1, visual_mod_id=visual_mod_id,
                                                                visual_modulation_params=visual_modulator_params,
                                                                spatial_modulation_params=conv1_att,
                                                                visual_modulation=use_visual_modulator and mod_early_conv,
                                                                spatial_modulation=use_spatial_modulator and mod_early_conv)

                    net_2 = slim.max_pool2d(net_1, [2, 2], scope='pool1')
                    net_2, visual_mod_id = modulated_conv_block(net_2, 2, 128,
                                                                scope_id=2, visual_mod_id=visual_mod_id,
                                                                visual_modulation_params=visual_modulator_params,
                                                                spatial_modulation_params=conv2_att,
                                                                visual_modulation=use_visual_modulator and mod_early_conv,
                                                                spatial_modulation=use_spatial_modulator and mod_early_conv)

                    net_3 = slim.max_pool2d(net_2, [2, 2], scope='pool2')
                    net_3, visual_mod_id = modulated_conv_block(net_3, 3, 256,
                                                                scope_id=3, visual_mod_id=visual_mod_id,
                                                                visual_modulation_params=visual_modulator_params,
                                                                spatial_modulation_params=conv3_att,
                                                                visual_modulation=use_visual_modulator,
                                                                spatial_modulation=use_spatial_modulator)
                    net_4 = slim.max_pool2d(net_3, [2, 2], scope='pool3')
                    net_4, visual_mod_id = modulated_conv_block(net_4, 3, 512,
                                                                scope_id=4, visual_mod_id=visual_mod_id,
                                                                visual_modulation_params=visual_modulator_params,
                                                                spatial_modulation_params=conv4_att,
                                                                visual_modulation=use_visual_modulator,
                                                                spatial_modulation=use_spatial_modulator)
                    net_5 = slim.max_pool2d(net_4, [2, 2], stride=1, scope='pool4')
                    net_5, visual_mod_id = modulated_conv_block(net_5, 3, 512,
                                                                dilation=2, scope_id=5, visual_mod_id=visual_mod_id,
                                                                visual_modulation_params=visual_modulator_params,
                                                                spatial_modulation_params=conv5_att,
                                                                visual_modulation=use_visual_modulator,
                                                                spatial_modulation=use_spatial_modulator)
                    pool5 = slim.max_pool2d(net_5, [3, 3], stride=1, scope='pool5')
                    ## hole = 6
                    fc6_1 = slim.conv2d(pool5, 1024, [3, 3], rate=6, scope='fc6_1')
                    fc6_1 = slim.dropout(fc6_1, 0.5, is_training=is_training, scope='drop6_1')
                    fc7_1 = slim.conv2d(fc6_1, 1024, [1, 1], scope='fc7_1')
                    fc7_1 = slim.dropout(fc7_1, 0.5, is_training=is_training, scope='drop7_1')
                    fc8_voc12_1 = slim.conv2d(fc7_1, 1, [1, 1], activation_fn=None, scope='fc8_voc12_1')
                    ## hole = 12
                    fc6_2 = slim.conv2d(pool5, 1024, [3, 3], rate=12, scope='fc6_2')
                    fc6_2 = slim.dropout(fc6_2, 0.5, is_training=is_training, scope='drop6_2')
                    fc7_2 = slim.conv2d(fc6_2, 1024, [1, 1], scope='fc7_2')
                    fc7_2 = slim.dropout(fc7_2, 0.5, is_training=is_training, scope='drop7_2')
                    fc8_voc12_2 = slim.conv2d(fc7_2, 1, [1, 1], activation_fn=None, scope='fc8_voc12_2')
                    ## hole = 18
                    fc6_3 = slim.conv2d(pool5, 1024, [3, 3], rate=18, scope='fc6_3')
                    fc6_3 = slim.dropout(fc6_3, 0.5, is_training=is_training, scope='drop6_3')
                    fc7_3 = slim.conv2d(fc6_3, 1024, [1, 1], scope='fc7_3')
                    fc7_3 = slim.dropout(fc7_3, 0.5, is_training=is_training, scope='drop7_3')
                    fc8_voc12_3 = slim.conv2d(fc7_3, 1, [1, 1], activation_fn=None, scope='fc8_voc12_3')
                    ## hole = 24
                    fc6_4 = slim.conv2d(pool5, 1024, [3, 3], rate=24, scope='fc6_4')
                    fc6_4 = slim.dropout(fc6_4, 0.5, is_training=is_training, scope='drop6_4')
                    fc7_4 = slim.conv2d(fc6_4, 1024, [1, 1], scope='fc7_4')
                    fc7_4 = slim.dropout(fc7_4, 0.5, is_training=is_training, scope='drop7_4')
                    fc8_voc12_4 = slim.conv2d(fc7_4, 1, [1, 1], activation_fn=None, scope='fc8_voc12_4')
                    fc8_voc12 = fc8_voc12_1 + fc8_voc12_2 + fc8_voc12_3 + fc8_voc12_4
                    with slim.arg_scope([slim.conv2d_transpose],
                                        activation_fn=None, biases_initializer=None, padding='VALID',
                                        trainable=False):
                        score_full = slim.conv2d_transpose(fc8_voc12, 1, 16, 8, scope='score-up')
                    net = crop_features(score_full, im_size)
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points


def test(dataset, model_params, checkpoint_file, result_path, batch_size=1, config=None):
    tf.logging.set_verbosity(tf.logging.INFO)
    assert batch_size == 1, "only allow batch size equal to 1 for testing"
    # Input data
    guide_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])
    gb_image = tf.placeholder(tf.float32, [batch_size, None, None, 1])
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])

    # Create model
    # split the model into visual modulator and other parts, visual modulator only need to run once
    if model_params.use_visual_modulator:
        v_m_params = visual_modulator(guide_image, model_params, is_training=False)

    net, end_points = osmn_deeplab([guide_image, gb_image, input_image], model_params,
                                   visual_modulator_params=v_m_params,
                                   is_training=False)
    probabilities = tf.nn.sigmoid(net)

    # Create a saver to load the network
    saver = tf.train.Saver([v for v in tf.global_variables()])  # if '-up' not in v.name and '-cr' not in v.name])

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)
        sess.run(interp_surgery(tf.global_variables()))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        print('start testing process')
        time_start = time.time()
        print("test size is: " + str(dataset.get_test_size()))
        for frame in range(dataset.get_test_size()):
            guide_images, gb_images, images, save_names = dataset.next_batch(batch_size, 'test')
            # create folder for results
            if len(save_names[0].split(os.sep)) > 1:
                save_path = os.path.join(result_path, *(save_names[0].split(os.sep)[:-1]))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            if images is None or gb_images is None:
                # first frame of a sequence
                if model_params.use_visual_modulator:
                    curr_v_m_params = sess.run(v_m_params, feed_dict={guide_image: guide_images})
                # create a black dummy image for result of the first frame, to be compatible with DAVIS eval toolkit
                scipy.misc.imsave(os.path.join(result_path, save_names[0]), np.zeros(guide_images.shape[1:3]))
            else:
                feed_dict = {gb_image: gb_images, input_image: images}
                if model_params.use_visual_modulator:
                    feed_dict[v_m_params] = curr_v_m_params
                res_all = sess.run([probabilities], feed_dict=feed_dict)
                res = res_all[0]
                res_np = res.astype(np.float32)[:, :, :, 0] > 0.5
                print('Saving ' + os.path.join(result_path, save_names[0]))
                scipy.misc.imsave(os.path.join(result_path, save_names[0]), res_np[0].astype(np.float32))
                curr_score_name = save_names[0][:-4]
                if model_params.save_score:
                    print('Saving ' + os.path.join(result_path, curr_score_name) + '.npy')
                    np.save(os.path.join(result_path, curr_score_name), res.astype(np.float32)[0, :, :, 0])
        time_finish = time.time()
        time_elapsed = time_finish - time_start
        print('Total time elapsed: %.3f seconds' % time_elapsed)
        print('Each frame takes %.3f seconds' % (time_elapsed / dataset.get_test_size()))


# Set deconvolutional layers to compute bilinear interpolation
def interp_surgery(variables):
    interp_tensors = []
    for v in variables:
        if '-up' in v.name:
            h, w, k, m = v.get_shape()
            tmp = np.zeros((m, k, h, w))
            if m != k:
                raise Exception('input + output channels need to be the same')

            if h != w:
                raise Exception('filters need to be square')
            up_filter = upsample_filt(int(h))
            tmp[list(range(m)), list(range(k)), :, :] = up_filter
            interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
    return interp_tensors


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


'''finished model construction'''
import sys, argparse, common_args, json
from PIL import Image
from scipy import ndimage
import random
import multiprocessing as mp
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax
from util import get_mask_bbox, get_gb_image, to_bgr, mask_image, data_augmentation, \
    get_dilate_structure, perturb_mask


def _get_obj_mask(image, idx):
    return Image.fromarray((np.array(image) == idx).astype(np.uint8))


def get_one(sample, new_size, args):
    if len(sample) == 4:
        # guide image is both for appearance and location guidance
        guide_image = Image.open(sample[0])
        guide_label = Image.open(sample[1])
        image = Image.open(sample[2])
        label = Image.open(sample[3])
        ref_label = guide_label
    else:
        # guide image is only for appearance guidance, ref label is only for location guidance
        guide_image = Image.open(sample[0])
        guide_label = Image.open(sample[1])
        # guide_image = Image.open(sample[2])
        ref_label = Image.open(sample[2])
        image = Image.open(sample[3])
        label = Image.open(sample[4])
    if len(sample) > 5:
        label_id = sample[5]
    else:
        label_id = 0
    image = image.resize(new_size, Image.BILINEAR)
    label = label.resize(new_size, Image.NEAREST)
    ref_label = ref_label.resize(new_size, Image.NEAREST)
    guide_label = guide_label.resize(guide_image.size, Image.NEAREST)

    if label_id > 0:
        guide_label = _get_obj_mask(guide_label, label_id)
        ref_label = _get_obj_mask(ref_label, label_id)
        label = _get_obj_mask(label, label_id)
    guide_label_data = np.array(guide_label)
    bbox = get_mask_bbox(guide_label_data)  # return 4 max/min to represent bbox
    guide_image = guide_image.crop(bbox)
    guide_label = guide_label.crop(bbox)
    guide_image, guide_label = data_augmentation(guide_image, guide_label,
                                                 args.guide_size, data_aug_flip=args.data_aug_flip,
                                                 keep_aspect_ratio=args.vg_keep_aspect_ratio,
                                                 random_crop_ratio=args.vg_random_crop_ratio,
                                                 random_rotate_angle=args.vg_random_rotate_angle,
                                                 color_aug=args.vg_color_aug)
    if not args.use_original_mask:
        gb_image = get_gb_image(np.array(ref_label), center_perturb=args.sg_center_perturb_ratio,
                                std_perturb=args.sg_std_perturb_ratio)
    else:
        gb_image = perturb_mask(np.array(ref_label))
        gb_image = ndimage.morphology.binary_dilation(gb_image,
                                                      structure=args.dilate_structure) * 255
    image_data = np.array(image, dtype=np.float32)
    label_data = np.array(label, dtype=np.uint8) > 0
    image_data = to_bgr(image_data)
    image_data = (image_data - args.mean_value) * args.scale_value
    guide_label_data = np.array(guide_label, dtype=np.uint8)
    guide_image_data = np.array(guide_image, dtype=np.float32)
    guide_image_data = to_bgr(guide_image_data)
    guide_image_data = (guide_image_data - args.mean_value) * args.scale_value
    guide_image_data = mask_image(guide_image_data, guide_label_data)
    return guide_image_data, gb_image, image_data, label_data


class Dataset:
    def __init__(self, train_list, test_list, args,
                 data_aug=False):
        """Initialize the Dataset object
        Args:
        train_list: TXT file or list with the paths of the images to use for training (Images must be between 0 and 255)
        test_list: TXT file or list with the paths of the images to use for testing (Images must be between 0 and 255)
        Returns:
        """
        # Define types of data augmentation
        random.seed(1234)
        self.args = args
        self.data_aug = data_aug
        self.data_aug_flip = data_aug
        self.args.data_aug_flip = data_aug
        self.data_aug_scales = args.data_aug_scales
        self.use_original_mask = args.use_original_mask
        self.vg_random_rotate_angle = args.vg_random_rotate_angle
        self.vg_random_crop_ratio = args.vg_random_crop_ratio
        self.vg_color_aug = args.vg_color_aug
        self.vg_keep_aspect_ratio = args.vg_keep_aspect_ratio
        self.vg_pad_ratio = args.vg_pad_ratio
        self.sg_center_perturb_ratio = args.sg_center_perturb_ratio
        self.sg_std_perturb_ratio = args.sg_std_perturb_ratio
        self.bbox_sup = args.bbox_sup
        self.multiclass = hasattr(args, 'data_version') and args.data_version == 2017 \
                          or hasattr(args, 'multiclass') and args.multiclass
        self.train_list = train_list
        self.test_list = test_list
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = len(train_list)
        print('#training samples', self.train_size)
        self.test_size = len(test_list)
        self.train_idx = np.arange(self.train_size)
        self.test_idx = np.arange(self.test_size)
        self.crf_infer_steps = 5
        self.args.dilate_structure = get_dilate_structure(5)
        np.random.shuffle(self.train_idx)
        self.size = args.im_size
        self.mean_value = args.mean_value  # np.array((104, 117, 123))
        self.scale_value = args.scale_value  # 0.00787 for mobilenet
        self.args.guide_size = (224, 224)
        if args.num_loader > 1:
            self.pool = mp.Pool(processes=args.num_loader)

    def __del__(self):
        if self.args.num_loader > 1:
            self.pool.close()
            self.pool.join()

    def next_batch(self, batch_size, phase):
        """Get next batch of image (path) and labels
        Args:
        batch_size: Size of the batch
        phase: Possible options:'train' or 'test'
        Returns in training:
        images: Numpy arrays of the images
        labels: Numpy arrays of the labels
        Returns in testing:
        images: Numpy array of the images
        path: List of image paths
        """
        if phase == 'train':
            if self.train_ptr + batch_size <= self.train_size:
                idx = np.array(self.train_idx[self.train_ptr:self.train_ptr + batch_size])
                self.train_ptr += batch_size  # so train_ptr points to start of current batch
            else:
                # after one round on all train samples, shuffle and retrain
                np.random.shuffle(self.train_idx)  # shuffle a sequence in-place
                new_ptr = batch_size
                idx = np.array(self.train_idx[:new_ptr])
                self.train_ptr = new_ptr
            guide_images = []
            gb_images = []
            images = []
            labels = []
            if self.data_aug_scales:
                scale = random.choice(self.data_aug_scales)  # default 1
                new_size = (int(self.size[0] * scale), int(self.size[1] * scale))
            if self.args.num_loader == 1:
                batch = [get_one(self.train_list[i], new_size, self.args) for i in idx]
            else:
                batch = [self.pool.apply(get_one, args=(self.train_list[i], new_size, self.args)) for i in idx]
            for guide_image_data, gb_image, image_data, label_data in batch:
                guide_images.append(guide_image_data)
                gb_images.append(gb_image)
                images.append(image_data)
                labels.append(label_data)
            images = np.array(images)
            gb_images = np.array(gb_images)[..., np.newaxis]
            labels = np.array(labels)[..., np.newaxis]
            guide_images = np.array(guide_images)
            return guide_images, gb_images, images, labels
        elif phase == 'test':
            guide_images = []
            gb_images = []
            images = []
            image_paths = []
            self.crop_boxes = []
            self.images = []
            assert batch_size == 1, "Only allow batch size = 1 for testing"
            if self.test_ptr + batch_size < self.test_size:
                idx = np.array(self.test_idx[self.test_ptr:self.test_ptr + batch_size])
                self.test_ptr += batch_size  # test to where
            else:
                new_ptr = (self.test_ptr + batch_size) % self.test_size
                idx = np.hstack((self.test_idx[self.test_ptr:], self.test_idx[:new_ptr]))
                self.test_ptr = new_ptr
            i = idx[0]
            sample = self.test_list[i]
            if len(sample) > 4:
                label_id = sample[4]
            else:
                label_id = 0

            if sample[0] == None:
                # visual guide image / mask is none, only read spatial guide and input image
                first_frame = False
                ref_label = Image.open(sample[2])
                image = Image.open(sample[3])
                frame_name = sample[3].split(os.sep)[-1].split('.')[0] + '.png'
                if len(sample) > 5:
                    # vid_path/label_id/frame_name
                    ref_name = os.path.join(sample[5], frame_name)
                elif self.multiclass:
                    # seq_name/label_id/frame_name
                    ref_name = os.path.join(*(sample[2].split(os.sep)[-3:-1] + [frame_name]))
                else:
                    # seq_name/frame_name
                    ref_name = os.path.join(sample[2].split(os.sep)[-2], frame_name)
            else:
                # only process visual guide image / mask
                first_frame = True
                guide_image = Image.open(sample[0])
                guide_label = Image.open(sample[1])
                if len(sample) > 5:
                    # vid_path/label_id/frame_name
                    ref_name = os.path.join(sample[5], sample[1].split(os.sep)[-1])
                elif self.multiclass:
                    # seq_name/label_id/frame_name
                    ref_name = os.path.join(*(sample[1].split(os.sep)[-3:]))
                else:
                    # seq_name/frame_name
                    ref_name = os.path.join(*(sample[1].split(os.sep)[-2:]))
            if not first_frame:
                if len(self.size) == 2:
                    self.new_size = self.size
                else:
                    # resize short size of image to self.size[0]
                    resize_ratio = max(float(self.size[0]) / image.size[0], float(self.size[0]) / image.size[1])
                    self.new_size = (int(resize_ratio * image.size[0]), int(resize_ratio * image.size[1]))
                ref_label = ref_label.resize(self.new_size, Image.NEAREST)
                if label_id > 0:
                    ref_label = _get_obj_mask(ref_label, label_id)
                ref_label_data = np.array(ref_label)
                image_ref_crf = image.resize(self.new_size, Image.BILINEAR)
                self.images.append(np.array(image_ref_crf))
                image = image.resize(self.new_size, Image.BILINEAR)
                if self.use_original_mask:
                    gb_image = ndimage.morphology.binary_dilation(ref_label_data,
                                                                  structure=self.args.dilate_structure) * 255
                else:
                    gb_image = get_gb_image(ref_label_data, center_perturb=0, std_perturb=0)
                image_data = np.array(image, dtype=np.float32)
                image_data = to_bgr(image_data)
                image_data = (image_data - self.mean_value) * self.scale_value
                gb_images.append(gb_image)
                images.append(image_data)
                images = np.array(images)
                gb_images = np.array(gb_images)[..., np.newaxis]
                guide_images = None
            else:
                # process visual guide images
                # resize to same size of guide_image first, in case of full resolution input
                guide_label = guide_label.resize(guide_image.size, Image.NEAREST)
                if label_id > 0:
                    guide_label = _get_obj_mask(guide_label, label_id)
                bbox = get_mask_bbox(np.array(guide_label))
                guide_image = guide_image.crop(bbox)
                guide_label = guide_label.crop(bbox)
                guide_image, guide_label = data_augmentation(guide_image, guide_label,
                                                             self.args.guide_size, data_aug_flip=False,
                                                             pad_ratio=self.vg_pad_ratio,
                                                             keep_aspect_ratio=self.vg_keep_aspect_ratio)

                guide_image_data = np.array(guide_image, dtype=np.float32)
                guide_image_data = to_bgr(guide_image_data)
                guide_image_data = (guide_image_data - self.mean_value) * self.scale_value
                guide_label_data = np.array(guide_label, dtype=np.uint8)
                if not self.bbox_sup:
                    guide_image_data = mask_image(guide_image_data, guide_label_data)
                guide_images.append(guide_image_data)
                guide_images = np.array(guide_images)
                images = None
                gb_images = None
            image_paths.append(ref_name)
            return guide_images, gb_images, images, image_paths
        else:
            return None, None, None, None

    def crf_processing(self, image, label, soft_label=False):
        crf = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
        if not soft_label:
            unary = unary_from_labels(label, 2, gt_prob=0.9, zero_unsure=False)
        else:
            if len(label.shape) == 2:
                p_neg = 1.0 - label
                label = np.concatenate((p_neg[..., np.newaxis], label[..., np.newaxis]), axis=2)
            label = label.transpose((2, 0, 1))
            unary = unary_from_softmax(label)
        crf.setUnaryEnergy(unary)
        crf.addPairwiseGaussian(sxy=(3, 3), compat=3)
        crf.addPairwiseBilateral(sxy=(40, 40), srgb=(5, 5, 5), rgbim=image, compat=10)
        crf_out = crf.inference(self.crf_infer_steps)

        # Find out the most probable class for each pixel.
        return np.argmax(crf_out, axis=0).reshape((image.shape[0], image.shape[1]))

    def get_train_size(self):
        return self.train_size

    def get_test_size(self):
        return self.test_size

    def train_img_size(self):
        return self.size

    def reset_idx(self):
        self.train_ptr = 0
        self.test_ptr = 0


def add_arguments(parser):
    group = parser.add_argument_group('Additional params')
    group.add_argument(
        '--data_path',
        type=str,
        required=False,
        default='/raid/ljyang/data/YoutubeVOS',
        help='Path to YoutubeVOS dataset')
    group.add_argument(
        '--vis_mod_model_path',
        type=str,
        required=False,
        default='models/vgg_16.ckpt',
        help='Model to initialize visual modulator')
    group.add_argument(
        '--seg_model_path',
        type=str,
        required=False,
        default='models/vgg_16.ckpt',
        help='Model to initialize segmentation model')
    group.add_argument(
        '--whole_model_path',
        type=str,
        required=False,
        default='',
        help='Source model path, could be a model pretrained on MS COCO')
    group.add_argument(
        '--randomize_guide',
        required=False,
        action='store_true',
        default=False,
        help='Whether to use randomized visual guide, or only the first frame')
    group.add_argument(
        '--label_valid_ratio',
        type=float,
        required=False,
        default=0.003,
        help='Parameter to search for valid visual guide, see details in code')
    group.add_argument(
        '--bbox_valid_ratio',
        type=float,
        required=False,
        default=0.2,
        help='Parameter to search for valid visual guide, see details in code')
    group.add_argument(
        '--test_split',
        type=str,
        required=False,
        default='valid',
        help='Which split to use for testing? val, train or test')
    group.add_argument(
        '--im_size',
        nargs=2, type=int,
        required=False,
        default=[448, 256],
        help='Input image size')
    group.add_argument(
        '--data_aug_scales',
        type=float, nargs='+',
        required=False,
        default=[1],
        help='Image scales to be used by data augmentation')
    group.add_argument(
        '--use_cached_list',
        action='store_true',
        default=False,
        help='Use cache train/test list')


def get_arguments():
    parser = argparse.ArgumentParser()
    common_args.add_arguments(parser)
    add_arguments(parser)
    args = parser.parse_args()

    return args


def predict(data_path, whole_model_path):
    # print(args)
    # sys.stdout.flush()
    args = get_arguments()
    baseDir = data_path
    random.seed(1234)

    val_path = os.path.join(baseDir, args.test_split, 'meta2.json')
    with open(val_path, 'r') as f:
        val_seqs = json.load(f)['videos']

    test_imgs_with_guide = []
    train_imgs_with_guide = []

    print("generating data list...")
    resDirLabel = args.result_path

    global fdhash
    fdhash = []
    for vid_id, seq in list(val_seqs.items()):  # iterate over videos
        # vid_id: "003234408d"
        fdhash.append(vid_id)
        vid_frames = seq['objects']  # object keys are like 1,2,3...
        vid_anno_path = os.path.join(baseDir, args.test_split, 'Annotations', vid_id)
        vid_image_path = os.path.join(baseDir, args.test_split, 'JPEGImages', vid_id)
        for label_id, obj_info in list(vid_frames.items()):  # iterate over objects in one video
            label_id = int(label_id)  # object keys are like 1,2,3...
            frames = obj_info['frames']  # [00000,00005,...
            # print frames
            res_fd = os.path.join(vid_id, str(label_id))
            # each sample:
            # visual guide image, visual guide mask, spatial guide mask, input image
            test_imgs_with_guide += [(os.path.join(vid_image_path, frames[0] + '.jpg'),
                                      os.path.join(vid_anno_path, frames[0] + '.png'),
                                      None, None, label_id, res_fd)]
            # reuse the visual modulation parameters
            # use predicted spatial guide image of former frame
            # skip the following if only one mask is labeled
            if len(frames) == 1: continue  # continues with the next iteration of the loop.
            # for frames after 00000, like 00005, need the spatial guide
            test_imgs_with_guide += [(None, None,
                                      os.path.join(vid_anno_path, frames[0] + '.png'),
                                      os.path.join(vid_image_path, frames[1] + '.jpg'), label_id, res_fd)]
            test_imgs_with_guide += [(None, None,
                                      os.path.join(resDirLabel, res_fd, prev_frame + '.png'),
                                      os.path.join(vid_image_path, frame + '.jpg'), 0, res_fd)
                                     for prev_frame, frame in zip(frames[1:-1], frames[2:])]

            # Define Dataset
    dataset = Dataset(train_imgs_with_guide, test_imgs_with_guide, args,
                      data_aug=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    # Predict the results
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(args.gpu_id)):
            checkpoint_path = whole_model_path
            test(dataset, args, checkpoint_path, args.result_path, config=config, batch_size=1)


def merge_masks(data_path, whole_model_path):
    args = get_arguments()
    # Merge the masks
    # Any PALETTE works
    PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0,
               191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]
    data_path = data_path
    pred_path = args.result_path
    dataset_split = args.test_split
    merge_path = args.result_path
    pred_size = 256
    listFile = '%s/%s/meta2.json' % (data_path, dataset_split)
    seq_data = json.load(open(listFile))['videos']
    im_num = 0
    iou = []
    seq_n = 0
    sample_n = 0
    prediction_size = (448, 256) if pred_size == 256 else (854, 480)
    subfd_names = []
    for vid_id, seq in seq_data.items():
        print('processing', vid_id)
        vid_frames = seq['objects']
        vid_anno_path = os.path.join(dataset_split, 'Annotations', vid_id)
        anno_path = os.path.join(data_path, vid_anno_path)
        anno_files = os.listdir(anno_path)
        sample_anno = Image.open(os.path.join(anno_path, anno_files[0]))
        width, height = sample_anno.size
        save_path = os.path.join(merge_path, vid_anno_path)
        # gather score and compute predicted label map
        frame_to_obj_dict = {}
        for label_id, obj_info in vid_frames.items():
            frames = obj_info['frames']
            for im_name in frames:
                if im_name in frame_to_obj_dict:
                    frame_to_obj_dict[im_name].append(int(label_id))
                else:
                    frame_to_obj_dict[im_name] = [int(label_id)]
        for im_name, obj_ids in frame_to_obj_dict.items():
            scores = []
            for label_id in obj_ids:
                score_path = os.path.join(pred_path, vid_id, str(label_id), im_name + '.npy')
                if not os.path.exists(score_path):
                    # no predicted score file, which means it is first frame for the
                    # corresponding object, read first frame gt label
                    gt = Image.open(os.path.join(anno_path, im_name + '.png'))
                    gt = gt.resize(prediction_size, Image.NEAREST)
                    gt = np.array(gt)
                    score = (gt == label_id).astype(np.float32)
                else:
                    path = os.path.join(pred_path, vid_id, str(label_id), im_name + '.npy')
                    with open(path, 'rb') as f:
                        score = np.load(f)
                        # score = np.load(open(os.path.join(pred_path, vid_id, str(label_id), im_name + '.npy')))

                # print(score.shape)
                scores.append(score)
            obj_ids_ext = np.array([0] + obj_ids, dtype=np.uint8)
            im_size = scores[0].shape
            bg_score = np.ones(im_size) * 0.5
            scores = [bg_score] + scores
            score_all = np.stack(tuple(scores), axis=-1)
            class_n = score_all.shape[2] - 1
            pred_idx = score_all.argmax(axis=2)
            label_pred = obj_ids_ext[pred_idx]

            res_im = Image.fromarray(label_pred, mode="P")
            res_im.putpalette(PALETTE)
            res_im = res_im.resize((width, height), Image.NEAREST)
            if not os.path.exists(os.path.join(save_path)):
                os.makedirs(os.path.join(save_path))
            res_im.save(os.path.join(save_path, im_name + '.png'))


# create video
import shutil
from tqdm import trange
from glob import glob
from collections import namedtuple
from moviepy.editor import ImageSequenceClip


def overlap(results_dir,
            masks_dir,
            images_dir,
            color_map,
            resize=False,
            image_file_extension='jpg',
            mask_file_extension='png',
            overwrite_existing=True):
    # Make a directory in which to store the results.
    if overwrite_existing and os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    mask_paths = glob(os.path.join(masks_dir, '*.' + mask_file_extension))
    num_images = len(mask_paths)

    print('The overlying images will be saved to "{}"'.format(results_dir))

    tr = trange(num_images, file=sys.stdout)
    tr.set_description('Processing images')

    for i in tr:
        maskpath = mask_paths[i]

        prediction = scipy.misc.imread(maskpath)
        if resize and not np.array_equal(prediction.shape[:2], resize):
            prediction = scipy.misc.imresize(prediction, resize)

        imagepath = os.path.join(images_dir, maskpath[-9:-3] + image_file_extension)
        image = scipy.misc.imread(imagepath)

        processed_image = np.asarray(
            print_segmentation_onto_image2(image=image, prediction=prediction), dtype=np.uint8)

        scipy.misc.imsave(os.path.join(results_dir, os.path.basename(maskpath)), processed_image)


def print_segmentation_onto_image2(image, prediction):
    image_size = image.shape

    # Create a template of shape `(image_height, image_width, 4)` to store RGBA values.
    mask = np.zeros(shape=(image_size[0], image_size[1], 4), dtype=np.uint8)
    mask[:, :, 3] = 127
    mask[:, :, :3] = prediction

    mask = scipy.misc.toimage(mask, mode="RGBA")

    output_image = scipy.misc.toimage(image)
    output_image.paste(mask, box=None,
                       mask=mask)
    return output_image


def create_video_from_images(video_output_name, image_input_dir,
                             frame_rate=30.0, image_file_extension='png'):
    image_paths = glob(os.path.join(image_input_dir, '*.' + image_file_extension))
    image_paths = sorted(image_paths)

    video = ImageSequenceClip(image_paths, fps=frame_rate)
    video.write_videofile("{}.mp4".format(video_output_name))


def get_results(data_path):
    args = get_arguments()
    Label = namedtuple('Label', [

        'name',  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class

        'id',  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.

        'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!

        'category',  # The name of the category that this label belongs to

        'categoryId',  # The ID of this category. Used to create ground truth images
        # on category level.

        'hasInstances',  # Whether this label distinguishes between single instances or not

        'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not

        'color',  # The color of this label
    ])
    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label('unlabeled', 0, 0, 'void', 0, False, True, (0, 0, 0)),
        Label('ego vehicle', 1, 0, 'void', 0, False, True, (0, 0, 0)),
        Label('rectification border', 2, 0, 'void', 0, False, True, (0, 0, 0)),
        Label('out of roi', 3, 0, 'void', 0, False, True, (0, 0, 0)),
        Label('static', 4, 0, 'void', 0, False, True, (0, 0, 0)),
        Label('dynamic', 5, 0, 'void', 0, False, True, (111, 74, 0)),
        Label('ground', 6, 0, 'void', 0, False, True, (81, 0, 81)),
        Label('road', 7, 1, 'flat', 1, False, False, (128, 64, 128)),
        Label('sidewalk', 8, 2, 'flat', 1, False, False, (244, 35, 232)),
        Label('parking', 9, 0, 'flat', 1, False, True, (250, 170, 160)),
        Label('rail track', 10, 0, 'flat', 1, False, True, (230, 150, 140)),
        Label('building', 11, 3, 'construction', 2, False, False, (70, 70, 70)),
        Label('wall', 12, 4, 'construction', 2, False, False, (102, 102, 156)),
        Label('fence', 13, 5, 'construction', 2, False, False, (190, 153, 153)),
        Label('guard rail', 14, 0, 'construction', 2, False, True, (180, 165, 180)),
        Label('bridge', 15, 0, 'construction', 2, False, True, (150, 100, 100)),
        Label('tunnel', 16, 0, 'construction', 2, False, True, (150, 120, 90)),
        Label('pole', 17, 6, 'object', 3, False, False, (153, 153, 153)),
        Label('polegroup', 18, 0, 'object', 3, False, True, (153, 153, 153)),
        Label('traffic light', 19, 7, 'object', 3, False, False, (250, 170, 30)),
        Label('traffic sign', 20, 8, 'object', 3, False, False, (220, 220, 0)),
        Label('vegetation', 21, 9, 'nature', 4, False, False, (107, 142, 35)),
        Label('terrain', 22, 10, 'nature', 4, False, False, (152, 251, 152)),
        Label('sky', 23, 11, 'sky', 5, False, False, (70, 130, 180)),
        Label('person', 24, 12, 'human', 6, True, False, (220, 20, 60)),
        Label('rider', 25, 13, 'human', 6, True, False, (255, 0, 0)),
        Label('car', 26, 14, 'vehicle', 7, True, False, (0, 0, 142)),
        Label('truck', 27, 15, 'vehicle', 7, True, False, (0, 0, 70)),
        Label('bus', 28, 16, 'vehicle', 7, True, False, (0, 60, 100)),
        Label('caravan', 29, 0, 'vehicle', 7, True, True, (0, 0, 90)),
        Label('trailer', 30, 0, 'vehicle', 7, True, True, (0, 0, 110)),
        Label('train', 31, 17, 'vehicle', 7, True, False, (0, 80, 100)),
        Label('motorcycle', 32, 18, 'vehicle', 7, True, False, (0, 0, 230)),
        Label('bicycle', 33, 19, 'vehicle', 7, True, False, (119, 11, 32)),
        Label('license plate', -1, 0, 'vehicle', 7, False, True, (0, 0, 142)),
    ]
    trainIds_to_colors_dict = {label.trainId: label.color for label in labels}
    TRAINIDS_TO_COLORS_DICT = trainIds_to_colors_dict
    TRAINIDS_TO_RGBA_DICT = {key: (*value, 127) for key, value in list(TRAINIDS_TO_COLORS_DICT.items())}

    mask_dir = []
    images_dir = []

    for fd_hash in fdhash:
        mask_path = os.path.join(args.result_path, 'valid/Annotations', str(fd_hash))
        mask_dir.append(mask_path)
        image_path = os.path.join(data_path, 'valid/JPEGImages', str(fd_hash))
        images_dir.append(image_path)

    for i, fd_hash in enumerate(fdhash): 
        overlap(results_dir=os.path.join('youtube_demo_video_images', fd_hash),
                masks_dir=mask_dir[i],
                images_dir=images_dir[i],
                color_map=TRAINIDS_TO_RGBA_DICT)

    for i, fd_hash in enumerate(fdhash):
        create_video_from_images(video_output_name='demo_video'+str(i),
                                 image_input_dir=os.path.join('youtube_demo_video_images', fd_hash),
                                 frame_rate=6.0,
                                 image_file_extension='png')


    from os import system
    for i in range(len(fdhash)):
      cmd = 'ffmpeg -i demo_video' + str(i) + '.mp4 -vf scale=448:256 demo' + str(i) +'.gif'
      system(cmd)
      cmd = 'rm -f demo_video' + str(i) + '.mp4'
      system(cmd)

