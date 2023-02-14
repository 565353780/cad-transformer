#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from tqdm import tqdm

from cad_transformer.Config.default import _C as config, update_config
from cad_transformer.Method.logger import create_logger
from cad_transformer.Model.cad_transformer import CADTransformer
from cad_transformer.Dataset.cad import CADDataset, CADDataLoader
from cad_transformer.Method.args import parse_args
from cad_transformer.Method.eval import do_eval, get_eval_criteria

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
distributed = num_gpus > 1


class Trainer(object):

    def __init__(self):
        return

    def train(self):
        args = parse_args()
        cfg = update_config(config, args)

        os.makedirs(cfg.log_dir, exist_ok=True)
        if cfg.eval_only:
            logger = create_logger(cfg.log_dir, 'val')
        elif cfg.test_only:
            logger = create_logger(cfg.log_dir, 'test')
        else:
            logger = create_logger(cfg.log_dir, 'train')

        # Distributed Train Config
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        device = torch.device('cuda:{}'.format(args.local_rank))

        # Create Model
        model = CADTransformer(cfg)
        CE_loss = torch.nn.CrossEntropyLoss().cuda()

        # Create Optimizer
        if cfg.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=cfg.learning_rate,
                                         betas=(0.9, 0.999),
                                         eps=1e-08,
                                         weight_decay=cfg.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=cfg.learning_rate,
                                        momentum=0.9)

        model = torch.nn.parallel.DistributedDataParallel(
            module=model.to(device),
            broadcast_buffers=False,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)
        model.train()

        # Load/Resume ckpt
        start_epoch = 0
        if cfg.load_ckpt != '':
            if os.path.exists(cfg.load_ckpt):
                checkpoint = torch.load(cfg.load_ckpt,
                                        map_location=torch.device("cpu"))
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                    cfg.load_ckpt, checkpoint['epoch']))
            else:
                logger.info("=>Failed: no checkpoint found at '{}'".format(
                    cfg.load_ckpt))
                exit(0)

        if cfg.resume_ckpt != '':
            if os.path.exists(cfg.load_ckpt):
                checkpoint = torch.load(cfg.resume_ckpt,
                                        map_location=torch.device("cpu"))
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint['epoch']
                logger.info(
                    f'=> resume checkpoint: {cfg.resume_ckpt} (epoch: {epoch})'
                )
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
            else:
                logger.info("=>Failed: no checkpoint found at '{}'".format(
                    cfg.resume_ckpt))
                exit(0)
        # Set up Dataloader
        torch.multiprocessing.set_start_method('spawn', force=True)
        val_dataset = CADDataset(split='val', do_norm=cfg.do_norm, cfg=cfg)
        val_dataloader = CADDataLoader(args.local_rank,
                                       dataset=val_dataset,
                                       batch_size=cfg.test_batch_size,
                                       shuffle=False,
                                       num_workers=cfg.WORKERS,
                                       drop_last=False)
        # Eval Only
        if args.local_rank == 0:
            if cfg.eval_only:
                eval_F1 = do_eval(model, val_dataloader, logger, cfg)
                exit(0)

        test_dataset = CADDataset(split='test', do_norm=cfg.do_norm, cfg=cfg)
        test_dataloader = CADDataLoader(args.local_rank,
                                        dataset=test_dataset,
                                        batch_size=cfg.test_batch_size,
                                        shuffle=False,
                                        num_workers=cfg.WORKERS,
                                        drop_last=False)
        # Test Only
        if args.local_rank == 0:
            if cfg.test_only:
                eval_F1 = do_eval(model, test_dataloader, logger, cfg)
                exit(0)

        train_dataset = CADDataset(split='train', do_norm=cfg.do_norm, cfg=cfg)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True)
        train_dataloader = CADDataLoader(args.local_rank,
                                         dataset=train_dataset,
                                         sampler=train_sampler,
                                         batch_size=cfg.batch_size,
                                         num_workers=cfg.WORKERS,
                                         drop_last=True)

        def bn_momentum_adjust(m, momentum):
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(
                    m, torch.nn.BatchNorm1d):
                m.momentum = momentum

        best_F1, eval_F1 = 0, 0
        best_epoch = 0
        global_epoch = 0

        print("> start epoch", start_epoch)
        for epoch in range(start_epoch, cfg.epoch):
            logger.info(f"=> {cfg.log_dir}")

            logger.info("\n\n")
            logger.info(f'Epoch {global_epoch + 1} ({epoch + 1}/{cfg.epoch})')
            lr = max(
                cfg.learning_rate * (cfg.lr_decay**(epoch // cfg.step_size)),
                cfg.LEARNING_RATE_CLIP)
            if epoch <= cfg.epoch_warmup:
                lr = cfg.learning_rate_warmup

            logger.info(f'Learning rate: {lr}')
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            momentum = cfg.MOMENTUM_ORIGINAL * (cfg.MOMENTUM_DECCAY
                                                **(epoch // cfg.step_size))
            if momentum < 0.01:
                momentum = 0.01
            logger.info(f'BN momentum updated to: {momentum}')
            model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
            model = model.train()

            # training loops
            with tqdm(train_dataloader,
                      total=len(train_dataloader),
                      smoothing=0.9) as _tqdm:
                for i, (image, xy, target, rgb_info, nns, offset_gt, inst_gt,
                        index, basename) in enumerate(_tqdm):
                    optimizer.zero_grad()

                    seg_pred = model(image, xy, rgb_info, nns)
                    seg_pred = seg_pred.contiguous().view(
                        -1, cfg.num_class + 1)
                    target = target.view(-1, 1)[:, 0]

                    loss_seg = CE_loss(seg_pred, target)
                    loss = loss_seg
                    loss.backward()
                    optimizer.step()
                    _tqdm.set_postfix(loss=loss.item(), l_seg=loss_seg.item())

                    if i % args.log_step == 0 and args.local_rank == 0:
                        logger.info(
                            f'Train loss: {round(loss.item(), 5)}, loss seg: {round(loss_seg.item(), 5)})'
                        )

            # Save last
            if args.local_rank == 0:
                logger.info('Save last model...')
                savepath = os.path.join(cfg.log_dir, 'last_model.pth')
                state = {
                    'epoch': epoch,
                    'best_F1': best_F1,
                    'best_epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            # assert validation?
            eval = get_eval_criteria(epoch)

            if args.local_rank == 0:
                if eval:
                    logger.info('> do validation')
                    eval_F1 = do_eval(model, val_dataloader, logger, cfg)
            # Save ckpt
            if args.local_rank == 0:
                if eval_F1 > best_F1:
                    best_F1 = eval_F1
                    best_epoch = epoch
                    logger.info(
                        f'Save model... Best F1:{best_F1}, Best Epoch:{best_epoch}'
                    )
                    savepath = os.path.join(cfg.log_dir, 'best_model.pth')
                    state = {
                        'epoch': epoch,
                        'best_F1': best_F1,
                        'best_epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)

            global_epoch += 1
        return True