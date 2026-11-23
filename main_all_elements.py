import os
import sys
import copy
import torch
import argparse
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from torch.cuda.amp import GradScaler
from module.DS_Backbone import LU_M2SR
from module.DynamicsCorrect import MultiVariableFusionWithWindCorrection
from config.config_setting_111 import *
from trainer.train_all import Trainer_areaAvg,Trainer_bicubic
from utils.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"

from config.config_setting_111 import *
from data_loader.dataloader import GetLoader,GetLoader2
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")


def main(config, par):
    print("~~~~~~~~~~~~~~~~current scale is: X{} ~~~~~~~~~~~~~~~~~~~~~".format(config.scale))
    print('#----------config----------#')
    print(config)
    print(par)
    print('#----------Creating logger----------#')

    writer = SummaryWriter(par.loss_dir)

    global logger
    logger_correct = get_logger('train_correct', par.log_dir_correct)
    logger_upsample = get_logger('train_upsample', par.log_dir_upsample)
    log_config_info(config, logger_correct)

    print('#----------GPU init----------#')

    n_gpu = torch.cuda.device_count()
    print(f"number of gpu is: {n_gpu}")
    set_seed(config.seed)
    torch.cuda.empty_cache()
    device = torch.device('cuda')


    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config

    c_list = [64, 128, 256, 512]

    logger_correct.info(f'c_list: {c_list}')
    model_correct = MultiVariableFusionWithWindCorrection(
        precip_channels=model_cfg['correct_precip_channels'],
        other_channels=model_cfg['correct_other_channels'],
        wind_channels=model_cfg['correct_uv_channels']
    ).to(torch.device('cuda'))
    model_upsampler = LU_M2SR(input_channels=model_cfg['upsampler_input_channels'],
                              out_channels=model_cfg['num_classes'],
                              rs_factor=config.scale,
                              c_list=c_list,
                              res=model_cfg['residual'],
                              is_cls=model_cfg['cls'],
                              split_nums=model_cfg['splict_nums'],
                              atten_config=model_cfg['dic_atten'],
                              ssd_config=model_cfg['dic_ssd'],
                              dic_geo_lsm=model_cfg['dic_geo_lsm'],
                              act_name='relu').to(torch.device('cuda'))
    # if n_gpu>1:
    #
    #     # model = torch.nn.DataParallel(model).to('cuda')
    #     model = torch.nn.DataParallel(model.to(device),output_device=gpu_ids[0])
    # else:
    #     model.to('cuda')
    # model._init_weights()

    print('#----------test params and flops----------#')
    # input_size = (3,36,36)
    # cal_params_flops(model,input_size,logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    config.criterion_type = 'WeightL1Loss'
    if config.criterion_type == 'L1PSNRSSIMLoss':
        criterion_reg = L1PSNRSSIMLoss().to(torch.device('cuda'))
    elif config.criterion_type == 'L1SSIMLoss':
        criterion_reg = L1SSIMLoss().to(torch.device('cuda'))
    elif config.criterion_type == 'L1Loss':
        criterion_reg = nn.L1Loss().to(torch.device('cuda'))
    elif config.criterion_type == 'L2Loss':
        criterion_reg = nn.MSELoss().to(torch.device('cuda'))
    elif config.criterion_type == 'WeightL1Loss':
        criterion_reg = Weight_MAE().to(torch.device('cuda'))

    # criterion_cls = nn.CrossEntropyLoss().to(torch.device('cuda'))
    # bins = torch.Tensor([0.1, 2.5, 10, 50]).to(torch.device('cuda'))

    criterion_cls = FocalLoss().to(torch.device('cuda'))
    criterion_wavelet = SWTLoss().to(torch.device('cuda'))

    correct_loss = PhysicsConstrainedLoss()

    print(
        f'~~~~~~~~~~~~~~~~~[Current config] loss type: {config.criterion_type},  optimizer type: {config.opt}, scheduler type: {config.sch}~~~~~~~~~~~~~~~~~~~~')

    print('#----------Set other params----------#')

    # set early_stopping
    early_stopping = EarlyStopping(patience=10, min_delta=1e-8)

    print('#----------Training----------#')
    if par.downSample_mode=='areaAvg':
        print('#----------Preparing dataset----------#')
        train_data_path = os.path.join(config.train_data_path, f'train_{par.dataset_name}_2.hdf5')
        val_data_path = os.path.join(config.test_data_path, f'valid_{par.dataset_name}_2.hdf5')
        test_data_path = os.path.join(config.test_data_path, f'test_{par.dataset_name}_2.hdf5')

        train_loader = GetLoader2(train_data_path,
                                  batch_size=par.batch_size,
                                  num_workers=par.num_workers,
                                  shuffle_is_true=True)
        val_loader = GetLoader2(val_data_path,
                                batch_size=par.batch_size,
                                num_workers=par.num_workers,
                                shuffle_is_true=False)
        test_loader = GetLoader2(test_data_path,
                                 batch_size=1,
                                 num_workers=par.num_workers,
                                 shuffle_is_true=False)
        Model_train = Trainer_areaAvg(par,
                          config,
                          model_correct,
                          model_upsampler,
                          train_loader,
                          val_loader,
                          test_loader,
                          correct_loss,
                          criterion_reg,
                          criterion_cls,
                          logger_correct,
                          logger_upsample,
                          writer,
                          early_stopping_func=None
                          )

    elif par.downSample_mode == 'bicubic':
        print('#----------Preparing dataset----------#')
        train_data_path = os.path.join(config.train_data_path, f'train_{par.dataset_name}_4.hdf5')
        val_data_path = os.path.join(config.test_data_path, f'valid_{par.dataset_name}_4.hdf5')
        test_data_path = os.path.join(config.test_data_path, f'test_{par.dataset_name}_4.hdf5')

        train_loader = GetLoader2(train_data_path,
                                 batch_size=par.batch_size,
                                 num_workers=par.num_workers,
                                 shuffle_is_true=True)
        val_loader = GetLoader2(val_data_path,
                               batch_size=par.batch_size,
                               num_workers=par.num_workers,
                               shuffle_is_true=False)
        test_loader = GetLoader2(test_data_path,
                                batch_size=1,
                                num_workers=par.num_workers,
                                shuffle_is_true=False)
        Model_train = Trainer_bicubic(par,
                                      config,
                                      model_correct,
                                      model_upsampler,
                                      train_loader,
                                      val_loader,
                                      test_loader,
                                      correct_loss,
                                      criterion_reg,
                                      criterion_cls,
                                      logger_correct,
                                      logger_upsample,
                                      writer,
                                      early_stopping_func=None
                                      )

    return Model_train


if __name__ == '__main__':

    dataset_name = '8x_minmax'
    parser = argparse.ArgumentParser(description='DyCDer')

    parser.add_argument('--mode', type=str, default='111',
                        help='mode of the model')
    parser.add_argument('--split_num', type=int, default=8,
                        help='number of split channel')
    parser.add_argument('--downSample_mode', type=str, default='areaAvg',
                        help='mode of the downsample precipitation model,areaAvg,bilinear, nearest,bicubic，nearest_unclass')
    # Hardware specifications
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of threads for data loading')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training')
    # train setting
    parser.add_argument('--dataset_name', type=str, default=dataset_name,
                        help='dataset name')
    parser.add_argument('--data_norm_mode', type=str, default='norm',
                        help='data process mode')
    parser.add_argument('--is_sequential', action='store_true', default=True,
                        help='select fixed schduler')
    parser.add_argument('--gclip', type=float, default=0,
                        help='gradient clipping threshold (0 = no clipping)')

    parser.add_argument('--lr_statistics_path', type=str, default='data/era5.json',
                        help='lr dataset mean, std file')
    parser.add_argument('--hr_statistics_path', type=str, default='data/cwa.json',
                        help='hr dataset mean, std file')
    parser.add_argument('--train_indexs_path', type=str, default=f"data/index/train_index_8x_{dataset_name}.npy",
                        help='test dataset name')
    parser.add_argument('--valid_indexs_path', type=str, default=f"data/index/valid_index_{dataset_name}.npy",
                        help='lr dataset mean, std file')
    parser.add_argument('--test_indexs_path', type=str, default=f"data/index/test_index_{dataset_name}.npy",
                        help='hr dataset mean, std file')
    parser.add_argument('--loss_dir', type=str, default='',
                        help='loss tensorboard path')

    parser.add_argument('--log_dir_correct', type=str, default='',
                        help='train correct log path')
    parser.add_argument('--log_dir_upsample', type=str, default='',
                        help='train upsample log path')

    parser.add_argument('--ck_dir_correcter', type=str, default='',
                        help='checkpoint path for correcter')
    parser.add_argument('--ck_dir_upsampler', type=str, default='',
                        help='checkpoint path for upsampler')
    # Resume setting
    parser.add_argument('--resume', type=int, default=0,
                        help='')
    parser.add_argument('--resume_dir', type=str, default='DyCDer/x8/splict_nums_8/18_May_2026_20h_28m_36s/',
                        help='resume dir')

    # validation setting
    parser.add_argument('--save_interval', type=int, default=10,
                        help='save_interval for plt valid img')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='save_interval for valid')
    parser.add_argument('--save_valid_img_path_correcter', type=str, default='',
                        help='save sr image for correcter')
    parser.add_argument('--save_valid_img_path_upsampler', type=str, default='',
                        help='save sr image for upsampler')
    # test
    parser.add_argument('--test_correcter_only', action='store_true', default=False,
                        help='set this option to test the correct model')
    parser.add_argument('--test_upsampler', action='store_true', default=False,
                        help='set this option to test the upsample model')
    parser.add_argument('--test_root_path', type=str,
                        default='DyCDer/x8/splict_nums_8/12_May_2026_23h_30m_10s/',
                        help='root path for test,must write this')
    parser.add_argument('--save_test_img_path_correcter', type=str, default='',
                        help='save sr image for correcter')
    parser.add_argument('--save_test_img_path_upsampler', type=str, default='',
                        help='save sr image for upsampler')
    parser.add_argument('--save_img_data_correcter', type=str, default='',
                        help='save sr data for correcter')
    parser.add_argument('--save_img_data_upsampler', type=str, default='',
                        help='save sr data for upsampler')
    parser.add_argument('--save_gt_path', type=str, default='',
                        help='save low-resolution and high-resolution images together')
    parser.add_argument('--save_results_path_correcter', type=str, default='',
                        help='save results for correcter')
    parser.add_argument('--save_results_path_upsampler', type=str, default='',
                        help='save results for upsampler')


    par = parser.parse_args()

    if par.mode == '111':
        config = setting_111_split8_x8
    else:
        print("Error Mode, must be '000', '001', '101', '100', '011', '110', '101', or '111' !!!")

    if not par.test_correcter_only and not par.test_upsampler:
        if par.resume == 1:
            work_dir = par.resume_dir
            par.ck_dir_correcter = os.path.join(work_dir, 'train', 'checkpoint', 'correcter')
            par.ck_dir_upsampler = os.path.join(work_dir, 'train', 'checkpoint', 'upsampler')
            par.loss_dir = os.path.join(work_dir, 'train', 'loss_log')

            valid_dir_correcter = os.path.join(work_dir, 'valid', 'correcter')
            valid_dir_upsampler = os.path.join(work_dir, 'valid', 'upsampler')
            par.save_valid_img_path_correcter = os.path.join(valid_dir_correcter, 'img')
            par.save_valid_img_path_upsampler = os.path.join(valid_dir_upsampler, 'img')
            par.log_dir_correct = os.path.join(work_dir, 'train', 'log','correct')
            par.log_dir_upsample = os.path.join(work_dir, 'train', 'log','upsample')

        elif par.resume == 0:
            work_dir = config.work_dir
            par.ck_dir_correcter = os.path.join(work_dir, 'train', 'checkpoint', 'correcter')
            par.ck_dir_upsampler = os.path.join(work_dir, 'train', 'checkpoint', 'upsampler')
            par.loss_dir = os.path.join(work_dir, 'train', 'loss_log')

            valid_dir_correcter = os.path.join(work_dir, 'valid', 'correcter')
            valid_dir_upsampler = os.path.join(work_dir, 'valid', 'upsampler')
            par.save_valid_img_path_correcter = os.path.join(valid_dir_correcter, 'img')
            par.save_valid_img_path_upsampler = os.path.join(valid_dir_upsampler, 'img')
            par.log_dir_correct = os.path.join(work_dir, 'train', 'log', 'correct')
            par.log_dir_upsample = os.path.join(work_dir, 'train', 'log', 'upsample')

        else:
            raise ValueError('par.resume should be 0 or 1 !!!')

        os.makedirs(par.loss_dir, exist_ok=True)
        os.makedirs(par.save_valid_img_path_correcter, exist_ok=True)
        os.makedirs(par.save_valid_img_path_upsampler, exist_ok=True)
        os.makedirs(par.ck_dir_correcter, exist_ok=True)
        os.makedirs(par.ck_dir_upsampler, exist_ok=True)
        os.makedirs(par.log_dir_correct, exist_ok=True)
        os.makedirs(par.log_dir_upsample, exist_ok=True)


    else:
        par.save_img_data_correcter = os.path.join(par.test_root_path, 'test/img_data/')
        par.save_test_img_path_correcter = os.path.join(par.test_root_path, 'test/img/')
        par.save_img_data_upsampler = os.path.join(par.test_root_path, 'test/img_data/')
        par.save_test_img_path_upsampler = os.path.join(par.test_root_path, 'test/img/')
        par.ck_dir_upsampler = os.path.join(par.test_root_path, 'train', 'checkpoint', 'upsampler')
        par.ck_dir_correcter = os.path.join(par.test_root_path, 'train', 'checkpoint', 'correcter')
        par.save_results_path_correcter = os.path.join(par.test_root_path, 'test/results/correct/')
        par.save_results_path_upsampler = os.path.join(par.test_root_path, 'test/results/upsample')
        par.log_dir_correct = os.path.join(par.test_root_path, 'test', 'log', 'correct')
        par.log_dir_upsample = os.path.join(par.test_root_path, 'test', 'log', 'upsample')

        os.makedirs(par.log_dir_correct, exist_ok=True)
        os.makedirs(par.log_dir_upsample, exist_ok=True)
        os.makedirs(par.save_img_data_correcter, exist_ok=True)
        os.makedirs(par.save_test_img_path_correcter, exist_ok=True)
        os.makedirs(par.save_img_data_upsampler, exist_ok=True)
        os.makedirs(par.save_test_img_path_upsampler, exist_ok=True)
        os.makedirs(par.save_results_path_correcter, exist_ok=True)
        os.makedirs(par.save_results_path_upsampler, exist_ok=True)

    config.num_workers = par.num_workers
    config.batch_size = par.batch_size

    Model_train = main(config, par)
    if par.test_correcter_only:
        Model_train.test_correct()
        print(f"test batch size is: {par.batch_size}")
    elif par.test_upsampler:
        Model_train.test_upsample()
        print(f"test batch size is: {par.batch_size}")
    else:
        Model_train.train()


