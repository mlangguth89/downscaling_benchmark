import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('-stat_dir', type=str,
                        help='The path to the training statistic json file directory (used for denormalise output)')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None


    for phase, dataset_opt in opt['datasets'].items():
        ## dataset for celebahq
        if opt["datasets"][phase]["name"] != "Precipitation":
            if phase == 'train' and args.phase != 'val':
                train_set = Data.create_dataset(dataset_opt, phase)
                train_loader = Data.create_dataloader(
                    train_set, dataset_opt, phase)
            elif phase == 'val':
                val_set = Data.create_dataset(dataset_opt, phase)
                val_loader = Data.create_dataloader(
                    val_set, dataset_opt, phase)
        else:
            if phase == 'train' and args.phase != 'val':
                train_loader = Data.create_loader_prep(file_path = opt["datasets"][phase]["dataroot"], batch_size=opt["datasets"][phase]["batch_size"], k=0.01, mode=phase)
            elif phase == 'val':
                val_loader = Data.create_loader_prep(
                    file_path = opt["datasets"][phase]["dataroot"], k=0.01, mode=phase, stat_path=args.stat_dir)



    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                #remove th following two lines if the dataset is not precipipation
                train_data["HR"] = torch.transpose(train_data["HR"],0,1)
                train_data["SR"] = train_data["SR"][0,...]
                #print("train_data", train_data["HR"].shape)
                #print("LR", train_data["LR"].shape)
                #print("SR", train_data["SR"].shape) #[4,8,160,160]

                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                #if current_step % opt['train']['val_freq'] == 0:
                #    avg_psnr = 0.0
                #    idx = 0
                #    result_path = '{}/{}'.format(opt['path']
                #                                 ['results'], current_epoch)
                #    os.makedirs(result_path, exist_ok=True)

                #    diffusion.set_new_noise_schedule(
                #        opt['model']['beta_schedule']['val'], schedule_phase='val')
                #    for _,  val_data in enumerate(val_loader):
                        
                        #if the dataset is precipitation, we need to remvoe the following two lines
                #        val_data["HR"] = torch.transpose(val_data["HR"],0,1)
                #        val_data["SR"] = val_data["SR"][0,...]
                #        print("valdiationd HR shpae", val_data["HR"].shape) #[4, 1, 160, 160]
                #        print("valdiationd SR shpae", val_data["SR"].shape ) #[4, 8, 160, 160]
                #        idx += 1
                #        diffusion.feed_data(val_data)
                #        diffusion.test(continous=False)
                #        visuals = diffusion.get_current_visuals()
                #        print("original shape of sr_img", Metrics.tensor2img(visuals['SR']).shape) #(160, 160)
                #        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                #        hr_img = Metrics.tensor2img(visuals['HR'][0,:,:])  # uint8
                #        #lr_img = Metrics.tensor2img(visuals['LR'][:,7,:,:,:])  # uint8
                #        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
                #        print("the shape of sr_img", sr_img.shape)
                #        print("The shape of HR", hr_img.shape)
                #        #print("The shape of lr",lr_img.shape)
                #        # generation
                #        #Metrics.save_img(
                #        #    hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                #        #Metrics.save_img(
                #        #    sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                #        #Metrics.save_img(
                #        #    lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                #        #Metrics.save_img(
                #        #    fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                #        #tb_logger.add_image(
                #        #    'Iter_{}'.format(current_step),
                #        #    np.transpose(np.concatenate(
                #        #        (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                #       #    idx)
                #        avg_psnr += Metrics.calculate_psnr(
                #            sr_img, hr_img)

                #        if wandb_logger:
                #            wandb_logger.log_image(
                #                f'validation_{idx}', 
                #                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                #            )

                #    avg_psnr = avg_psnr / idx
                #    diffusion.set_new_noise_schedule(
                #        opt['model']['beta_schedule']['train'], schedule_phase='train')
                #    # log
                #    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                #    logger_val = logging.getLogger('val')  # validation logger
                #    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                #        current_epoch, current_step, avg_psnr))
                #    # tensorboard logger
                #    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                #    if wandb_logger:
                #        wandb_logger.log_metrics({
                #            'validation/val_psnr': avg_psnr,
                #            'validation/val_step': val_step
                #        })
                #        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)
            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            val_data["HR"] = torch.transpose(val_data["HR"],0,1)
            val_data["SR"] = val_data["SR"][0,...]
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            #Denormalise dataset

            stat_file = os.path.join(args.stat_dir, "statistics.json")
            with open(stat_file, 'r') as f:
                stat_data = json.load(f)
            vars_out_patches_mean = stat_data['yw_hourly_tar_mean']
            vars_out_patches_std = stat_data['yw_hourly_tar_std']


            # griId img
            #print("visual SR",visuals['SR'].shape) #[44,1,160,160], [4,11,160,160]
            #print("visual LR", visuals["LR"].shape) #[4,8,16,16]
            #print("visual HR", visuals["HR"].shape) #[4, 1, 160, 160]
            #sr_img = Metrics.tensor2img(visuals['SR'])
            sr_img = Metrics.tensor2np(visuals['SR'],vars_out_patches_std,vars_out_patches_mean)
            hr_img = Metrics.tensor2np(val_data["HR"][:,0,:,:],vars_out_patches_std,vars_out_patches_mean)
            lr_img = Metrics.tensor2np(val_data['LR'][0,:,-1,:,:],vars_out_patches_std,vars_out_patches_mean)
            #sr_img = visuals['SR'].cpu().numpy()
            #hr_img = val_data["HR"][:,0,:,:].cpu().numpy()
            #lr_img = val_data['LR'][0,:,-1,:,:].cpu().numpy()
            #fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
            #print("sr_img shape",sr_img.shape) #[160,160,44]
            #print("hr_image.shape",hr_img.shape) #[4,160,160]

            Metrics.save_to_nc(
                sr_img, hr_img, lr_img, '{}/{}_{}_sr_process.nc'.format(result_path, current_step, idx))

            # generation
            # eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            # eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            #
            # avg_psnr += eval_psnr
            # avg_ssim += eval_ssim

            # if wandb_logger and opt['log_eval']:
            #     wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)
        #
        # avg_psnr = avg_psnr / idx
        # avg_ssim = avg_ssim / idx
        #
        # # log
        # logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        # logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        # logger_val = logging.getLogger('val')  # validation logger
        # logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
        #     current_epoch, current_step, avg_psnr, avg_ssim))
        #
        # if wandb_logger:
        #     if opt['log_eval']:
        #         wandb_logger.log_eval_table()
        #     wandb_logger.log_metrics({
        #         'PSNR': float(avg_psnr),
        #         'SSIM': float(avg_ssim)
        #     })
