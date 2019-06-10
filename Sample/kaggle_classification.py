# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 11:56:11 2018

@author: ALBERTO
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import os
import random
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Cropping2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
from collections import defaultdict
import pdb
import random
import imutils


random.seed(123)
N_Cls = 10
inDir = './kaggle'
DF = pd.read_csv(inDir + '/train_wkt_v4.csv')
GS = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
ISZ = 160
smooth = 1e-12


def _convert_coordinates_to_raster(coords, img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=GS, wkt_list_pandas=DF):
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask


def M(image_id):
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(inDir, 'sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img


def stretch_n(bands, lower_percent=5, higher_percent=95):
    out = np.zeros_like(bands)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    
    #pdb.set_trace()
    return out.astype(np.float32)


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def stick_all_train():
    print ("let's stick all imgs together")
    s = 835

    x = np.zeros((5 * s, 5 * s, 8))
    y = np.zeros((5 * s, 5 * s, N_Cls))
    #col_x = np.zeros((5 * s, 5 * s, 8))

    ids = sorted(DF.ImageId.unique())
    print( len(ids))
    for i in range(5):
        for j in range(5):
            id = ids[5 * i + j]

            img = M(id)
            img = stretch_n(img)
            #col_img = M(id)
            print( img.shape, id, np.amax(img), np.amin(img))
            x[s * i:s * i + s, s * j:s * j + s, :] = img[:s, :s, :]
            #col_x[s * i:s * i + s, s * j:s * j + s, :] = col_img[:s, :s, :]
            for z in range(N_Cls):
                y[s * i:s * i + s, s * j:s * j + s, z] = generate_mask_for_image_and_class(
                    (img.shape[0], img.shape[1]), id, z + 1)[:s, :s]
    #pdb.set_trace()
    print( np.amax(y), np.amin(y))

    np.save('./kaggle/x_trn_%d' % N_Cls, x)
    np.save('./kaggle/y_trn_%d' % N_Cls, y)
    #np.save('./kaggle/col_img', col_x)


def get_patches(img, msk, amt=10000, aug=True):
    is2 = int(1.0 * ISZ)
    xm, ym = img.shape[0] - is2, img.shape[1] - is2
    col = np.load('./kaggle/col_img.npy')
    
    x, y, x_col = [], [], []

    tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]
    for i in range(amt):
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)

        im = img[xc:xc + is2, yc:yc + is2]
        ms = msk[xc:xc + is2, yc:yc + is2]
        cc = col[xc:xc + is2, yc:yc + is2]

        for j in range(N_Cls):
            sm = np.sum(ms[:, :, j])
            if 1.0 * sm / is2 ** 2 > tr[j]:
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        im = im[::-1]
                        ms = ms[::-1]
                        cc = cc[::-1]
                    if random.uniform(0, 1) > 0.5:
                        im = im[:, ::-1]
                        ms = ms[:, ::-1]
                        cc = cc[:, ::-1]

                x.append(im)
                y.append(ms)
                x_col.append(cc)
                
    x, y = 2 * np.transpose(x, (0, 3, 1, 2)) - 1, np.transpose(y, (0, 3, 1, 2))
    #pdb.set_trace()
    print( x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y))
    np.save('./kaggle/trn_col', x_col)
    return x, y


def make_val():
    print( "let's pick some samples for validation")
    img = np.load('./kaggle/x_trn_%d.npy' % N_Cls)
    msk = np.load('./kaggle/y_trn_%d.npy' % N_Cls)
    x, y = get_patches(img, msk, amt=3000)

    np.save('./kaggle/x_tmp_%d' % N_Cls, x)
    np.save('./kaggle/y_tmp_%d' % N_Cls, y)


def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[3] - refer.get_shape()[3]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

'''
def get_unet(n_ch = 8,patch_height = 160, patch_width = 160):
    concat_axis = 1

    inputs = Input((n_ch, patch_width, patch_height))
    
    conv1 = Conv2D(32, (2, 2), padding="same", name="conv1_1", activation="relu", data_format="channels_first")(inputs)
    conv1 = Conv2D(32, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv1)
    conv2 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(pool1)
    conv2 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv2)

    conv3 = Conv2D(128, (2, 2), padding="same", activation="relu", data_format="channels_first")(pool2)
    conv3 = Conv2D(128, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv3)

    conv4 = Conv2D(256, (2, 2), padding="same", activation="relu", data_format="channels_first")(pool3)
    conv4 = Conv2D(256, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv4)

    conv5 = Conv2D(512, (2, 2), padding="same", activation="relu", data_format="channels_first")(pool4)
    conv5 = Conv2D(512, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv5)

    up_conv5 = UpSampling2D(size=(2, 2), data_format="channels_first")(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw), data_format="channels_first")(conv4)
    up6   = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = Conv2D(256, (2, 2), padding="same", activation="relu", data_format="channels_first")(up6)
    conv6 = Conv2D(256, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv6)

    up_conv6 = UpSampling2D(size=(2, 2), data_format="channels_first")(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw), data_format="channels_first")(conv3)
    up7   = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv2D(128, (2, 2), padding="same", activation="relu", data_format="channels_first")(up7)
    conv7 = Conv2D(128, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv7)

    up_conv7 = UpSampling2D(size=(2, 2), data_format="channels_first")(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw), data_format="channels_first")(conv2)
    up8   = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(up8)
    conv8 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv8)

    up_conv8 = UpSampling2D(size=(2, 2), data_format="channels_first")(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw), data_format="channels_first")(conv1)
    up9   = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D(32, (2, 2), padding="same", activation="relu", data_format="channels_first")(up9)
    conv9 = Conv2D(32, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9  = ZeroPadding2D(padding=(ch[0],cw[0]), data_format="channels_first")(conv9)
    conv10 = Conv2D(N_Cls, (1, 1), data_format="channels_first", activation="sigmoid")(conv9)
    
    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    print(model.summary())
    return model
'''


def calc_jacc(model):
    img = np.load('./kaggle/x_tmp_%d.npy' % N_Cls)
    msk = np.load('./kaggle/y_tmp_%d.npy' % N_Cls)

    prd = model.predict(img, batch_size=4)
    print( prd.shape, msk.shape)
    avg, trs = [], []

    for i in range(N_Cls):
        t_msk = msk[:, i, :, :]
        t_prd = prd[:, i, :, :]
        t_msk = t_msk.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])
        t_prd = t_prd.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])

        m, b_tr = 0, 0
        for j in range(10):
            tr = j / 10.0
            pred_binary_mask = t_prd > tr

            jk = jaccard_similarity_score(t_msk, pred_binary_mask)
            if jk > m:
                m = jk
                b_tr = tr
        print( i, m, b_tr)
        avg.append(m)
        trs.append(b_tr)

    score = sum(avg) / 10.0
    return score, trs


def mask_for_polygons(polygons, im_size):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def mask_to_polygons(mask, epsilon=5, min_area=1.):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def get_scalers(im_size, x_max, y_min):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    h, w = float(h), float(w)
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


def get_unet_64(n_ch = 8,patch_height = 160, patch_width = 160):
    concat_axis = 1

    inputs = Input((n_ch, patch_width, patch_height))
    
    conv1 = Conv2D(64, (2, 2), padding="same", name="conv1_1", activation="relu", data_format="channels_first")(inputs)
    conv1 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv1)
    conv2 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(pool1)
    conv2 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv2)

    conv3 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(pool2)
    conv3 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv3)

    conv4 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(pool3)
    conv4 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_first")(conv4)

    conv5 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(pool4)
    conv5 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv5)

    up_conv5 = UpSampling2D(size=(2, 2), data_format="channels_first")(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw), data_format="channels_first")(conv4)
    up6   = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(up6)
    conv6 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv6)

    up_conv6 = UpSampling2D(size=(2, 2), data_format="channels_first")(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw), data_format="channels_first")(conv3)
    up7   = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(up7)
    conv7 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv7)

    up_conv7 = UpSampling2D(size=(2, 2), data_format="channels_first")(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw), data_format="channels_first")(conv2)
    up8   = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(up8)
    conv8 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv8)

    up_conv8 = UpSampling2D(size=(2, 2), data_format="channels_first")(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw), data_format="channels_first")(conv1)
    up9   = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(up9)
    conv9 = Conv2D(64, (2, 2), padding="same", activation="relu", data_format="channels_first")(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9  = ZeroPadding2D(padding=(ch[0],cw[0]), data_format="channels_first")(conv9)
    conv10 = Conv2D(N_Cls, (1, 1), data_format="channels_first", activation="sigmoid")(conv9)
    
    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    print(model.summary())
    return model


def train_net():
    print( "start train net")
    x_val, y_val = np.load('./kaggle/x_tmp_%d.npy' % N_Cls), np.load('./kaggle/y_tmp_%d.npy' % N_Cls)
    print('x_val and y_val loaded')
    img = np.load('./kaggle/x_trn_%d.npy' % N_Cls)
    print('x_trn loaded')
    msk = np.load('./kaggle/y_trn_%d.npy' % N_Cls)
    print('y_trn loaded')

    x_trn, y_trn = get_patches(img, msk)
    print('got patches')
    
    '''
    img_Xtrn = x_trn[0]
    img_Ytrn = y_trn[0]
    img_Xval = x_val[0]
    img_Yval = y_val[0]
    for img_Xt in range(8):
        plt.imshow(img_Xtrn[img_Xt, :, :])
        plt.savefig('x_trn ' + str(img_Xt) + " .png")
        plt.close()
    for img_Yt in range(10):
        plt.imshow(img_Ytrn[img_Yt, :, :])
        plt.savefig('y_trn ' + str(img_Yt) + " .png")
        plt.close()
    for img_Xv in range(8):
        plt.imshow(img_Xval[img_Xv, :, :])
        plt.savefig('x_val ' + str(img_Xv) + " .png")
        plt.close()
    for img_Yv in range(10):
        plt.imshow(img_Yval[img_Yv, :, :])
        plt.savefig('y_val ' + str(img_Yv) + " .png")
        plt.close()
      
    val_img = y_val[0]
    val_img = np.rollaxis(val_img, 0, 3)
    for c in range(10):    
        class_ = val_img[:, :, c].sum()    
        print('The presence of class ' + str(c + 1) + ' is: ' + str((class_ / (160*160))*100) + '%')
    '''
    #pdb.set_trace()
    model = get_unet_64()
    print('got unet')
    #model.load_weights('weights/unet_10_jk0.7878')
    #model_checkpoint = ModelCheckpoint('weights/unet_tmp.hdf5', monitor='loss', save_best_only=True)
    for i in range(1):
        print(i)
        model.fit(x_trn, y_trn, batch_size = 4, nb_epoch = 5, verbose=1, shuffle=True,
                  validation_data=(x_val, y_val))
        
        del x_trn
        del y_trn
        
        output = model.predict(x_val)
        for img_out in range(10):
            plt.imshow(output[0, img_out, :, :])
            plt.savefig('output ' + str(img_out) + " .png")
            plt.close()
            
        #np.save('./kaggle/output', output)
        #x_trn, y_trn = get_patches(img, msk)
        #score, trs = calc_jacc(model)
        #print( 'val jk', score)
        model.save_weights('weights/unet_10_jk%.4f' % 1)

    return model, output


def predict_id(id, model, trs):
    img = M(id)
    x = stretch_n(img)

    cnv = np.zeros((960, 960, 8)).astype(np.float32)
    prd = np.zeros((N_Cls, 960, 960)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x

    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])

        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
        tmp = model.predict(x, batch_size=4)
        for j in range(tmp.shape[0]):
            prd[:, i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ] = tmp[j]

    # trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for i in range(N_Cls):
        prd[i] = prd[i] > trs[i]

    return prd[:, :img.shape[0], :img.shape[1]]


def predict_test(model, trs):
    print( "predict test")
    for i, id in enumerate(sorted(set(SB['ImageId'].tolist()))):
        msk = predict_id(id, model, trs)
        np.save('msk/10_%s' % id, msk)
        if i % 100 == 0: print( i, id)


def make_submit():
    print( "make submission file")
    df = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
    print( df.head())
    for idx, row in df.iterrows():
        id = row[0]
        kls = row[1] - 1

        msk = np.load('msk/10_%s.npy' % id)[kls]
        pred_polygons = mask_to_polygons(msk)
        x_max = GS.loc[GS['ImageId'] == id, 'Xmax'].as_matrix()[0]
        y_min = GS.loc[GS['ImageId'] == id, 'Ymin'].as_matrix()[0]

        x_scaler, y_scaler = get_scalers(msk.shape, x_max, y_min)

        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                      origin=(0, 0, 0))

        df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
        if idx % 100 == 0: print( idx)
    print( df.head())
    df.to_csv('subm/1.csv', index=False)


def check_predict(id='6120_2_3'):
    model = get_unet()
    model.load_weights('weights/unet_10_jk0.7878')

    msk = predict_id(id, model, [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1])
    img = M(id)

    plt.figure()
    ax1 = plt.subplot(131)
    ax1.set_title('image ID:6120_2_3')
    ax1.imshow(img[:, :, 5], cmap=plt.get_cmap('gist_ncar'))
    ax2 = plt.subplot(132)
    ax2.set_title('predict bldg pixels')
    ax2.imshow(msk[0], cmap=plt.get_cmap('gray'))
    ax3 = plt.subplot(133)
    ax3.set_title('predict bldg polygones')
    ax3.imshow(mask_for_polygons(mask_to_polygons(msk[0], epsilon=1), img.shape[:2]), cmap=plt.get_cmap('gray'))

    plt.show()



if __name__ == '__main__':
    stick_all_train()
    make_val()
    model, output = train_net()
#    score, trs = calc_jacc(model)
#    predict_test(model, trs)
#    make_submit()

    # bonus
#    check_predict()
    
    
    
# 5 epoch batch_size 32
# Tutti i conv2D a 64 
# val_acc: 0.9509
# jaccard_coef: 0.1086 - jaccard_coef_int: 0.0962
# val jk = 0.73


# 5 epoch batch_size 4
# conv2D base
# val_acc: 0.9551
# jaccard_coef: 0.0996 - jaccard_coef_int: 0.0888
# val jk = 0.74
    
# 5 epoch batch_size 4
# Tutti i conv2D a 64 
# val_acc: 0.9566
# jaccard_coef: 0.1039 - jaccard_coef_int: 0.0913
# val jk = 0.74
    
'''
The presence of class 1 is: 0.0%
The presence of class 2 is: 0.0%
The presence of class 3 is: 0.0%
The presence of class 4 is: 5.59375%
The presence of class 5 is: 34.40625%
The presence of class 6 is: 42.97265625%
The presence of class 7 is: 0.0%
The presence of class 8 is: 0.0%
The presence of class 9 is: 0.0%
The presence of class 10 is: 0.0%
'''

out = np.load('./kaggle/output.npy')
col_img_full = np.load('./kaggle/col_img.npy')
col = np.load('./kaggle/trn_col.npy')

def stretch2(band, lower_percent=2, higher_percent=98):
    a = 0 #np.min(band)
    b = 255  #np.max(band)
    c = np.percentile(band, lower_percent)
    d = np.percentile(band, higher_percent)        
    out = a + (band - c) * (b - a) / (d - c)    
    out[out<a] = a
    out[out>b] = b
    return out

def adjust_contrast(x):    
    for i in range(3):
        x[:,:,i] = stretch2(x[:,:,i])
    return x.astype(np.uint8)  


col_img = col[0]
c_img = np.zeros(shape=(160,160,3))
c_img[:,:,0] = col_img[:,:,4]
c_img[:,:,1] = col_img[:,:,2]
c_img[:,:,2] = col_img[:,:,1]
c_img = adjust_contrast(c_img).copy()
plt.imshow(c_img)
plt.savefig('trn_col.png')
plt.close()



##################
# Sliding Window #
##################

import slidingwindow as sw

x_val, y_val = np.load('./kaggle/x_tmp_%d.npy' % N_Cls), np.load('./kaggle/y_tmp_%d.npy' % N_Cls)
data = y_val[0]
data = np.rollaxis(data, 0, 3)
windows = sw.generate(data, sw.DimOrder.HeightWidthChannel, 16, 0)


window_dict = {}
wind = 1

for window in windows:
    subset = data[window.indices()]
    window_dict[wind] = {}
    for c in range(10):
        class_ = subset[:, :, c].sum()    
        #print('The presence of class ' + str(c + 1) + ' is: ' + str((class_ / (160*160))*100) + '%')
        window_dict[wind][c] = class_
        
    wind += 1

np.save('./kaggle/dict_windows', window_dict)

import numpy as np
#dict_w = np.load('./kaggle/windows.npy').item()
dict_w = np.load('./kaggle/dict_windows.npy').item()    

# Make a pie chart for every window (100 pie charts)
for w in window_dict:
    plt.pie(window_dict[w].values(), labels = window_dict[w].items())
    plt.title('Pie chart ' + str(w) + ' window')
    plt.savefig('window ' + str(w) + '.png')
    plt.close()

# Make a pie chart for the original picture
original = {}
for c in range(10):
    cha = data[:, :, c].sum()
    original[c] = cha

plt.pie(original.values(), labels = original.items())
plt.title('Pie chart original image')
plt.savefig('Original.png')
plt.close()


# Make a pie chart given by the sum of all the windows(no overlapping)
summed = {}
for window in windows:
    subset = data[window.indices()]
    for c in range(10):
        cha = subset[:, :, c].sum()
        if c not in summed.keys():
            summed[c] = cha
        else:
            summed[c] += cha

summed == original



