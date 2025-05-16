import logging
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from datasets.fundus.RIGA_dataloader import RIGA_labeled_set
from datasets.fundus.convert_csv_to_list import convert_labeled_list
from datasets.fundus.normalize import normalize_image
from datasets.fundus.transform import collate_fn_tr_styleaug, collate_fn_ts

from test import inference_riga
from utils import dice_coeff, bce_loss


def calc_riga_loss(outputs, label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    logits0 = outputs['masks'][:, 0]
    pred0 = torch.nn.Sigmoid()(logits0)
    label_batch0 = (label_batch[:, 0] > 0) * 1.0
    loss_ce0 = bce_loss(pred=pred0, label=label_batch0)
    loss_dice0 = dice_coeff(pred=pred0, label=label_batch0)
    loss0 = (1 - dice_weight) * loss_ce0 + dice_weight * loss_dice0

    logits1 = outputs['masks'][:, 1]
    pred1 = torch.nn.Sigmoid()(logits1)
    label_batch1 = (label_batch[:, 0] == 2) * 1.0
    loss_ce1 = bce_loss(pred=pred1, label=label_batch1)
    loss_dice1 = dice_coeff(pred=pred1, label=label_batch1)
    loss1 = (1 - dice_weight) * loss_ce1 + dice_weight * loss_dice1

    return loss0+loss1, loss_ce0+loss_ce1, loss_dice0+loss_dice1


def trainer_riga(args, model, snapshot_path, multimask_output):
    #single-source-domain
    source_csv = [os.path.join(args.root_path, args.Source_Dataset + '.csv')]

    tr_img_list, tr_label_list = convert_labeled_list(source_csv, r=1)


    print('Training Phase')
    print(source_csv)

    logging.basicConfig(filename=snapshot_path + "/prostate_RUNMC.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    tr_dataset = RIGA_labeled_set(args.root_path, tr_img_list, tr_label_list, img_normalize=False)
    trainloader = torch.utils.data.DataLoader(tr_dataset,
                                                batch_size=batch_size,
                                                num_workers=args.num_workers,
                                                shuffle=True,
                                                pin_memory=True,
                                                collate_fn=collate_fn_tr_styleaug)


    #to rest target domain
    target_name = args.Target_Dataset
    valid_loader = []
    for t_n in target_name:
        target_csv = [os.path.join(args.root_path, t_n + '.csv')]
        ts_img_list, ts_label_list = convert_labeled_list(target_csv, r=1)

        ts_dataset = RIGA_labeled_set(args.root_path, ts_img_list, ts_label_list)
        valid_loader.append(torch.utils.data.DataLoader(ts_dataset,
                                                    batch_size=1,
                                                    num_workers=args.num_workers//2,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    collate_fn=collate_fn_ts))


    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = bce_loss
    dice_loss = dice_coeff
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    # Initialize best scores for all metrics
    best_avg_score = 0.0
    best_epoch = 0
    best_scores = {
        'disc1': 0.0, 'cup1': 0.0,  # First target
        'disc2': 0.0, 'cup2': 0.0,  # Second target
        'disc3': 0.0, 'cup3': 0.0   # Third target
    }

    iterator = (range(max_epoch))
    for epoch_num in iterator:
        logging.info('Epoch %d / %d' % (epoch_num, max_epoch))
        for batch, data in enumerate(trainloader):
            x, y = data['data'], data['seg']
            x = torch.from_numpy(normalize_image(x)).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.float32)

            x, y = x.cuda(), y.cuda()

            outputs = model(x, multimask_output, args.img_size)

            loss, loss_ce, loss_dice = calc_riga_loss(outputs, y, ce_loss, dice_loss, args.dice_param)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, lr: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), lr_))

        save_interval = 20 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            #test
            mean_val_disc_dice1, mean_val_cup_dice1 = inference_riga(args=args, epoch=epoch_num, snapshot_path=snapshot_path, test_loader=valid_loader[0], model=model, test_save_path=None)
            writer.add_scalar('Base1_Disc_Dice', mean_val_disc_dice1, (epoch_num + 1) // save_interval)
            writer.add_scalar('Base1_Cup_Dice', mean_val_cup_dice1, (epoch_num + 1) // save_interval)
            
            mean_val_disc_dice2, mean_val_cup_dice2 = inference_riga(args=args, epoch=epoch_num, snapshot_path=snapshot_path, test_loader=valid_loader[1], model=model, test_save_path=None)
            writer.add_scalar('Base2_Disc_Dice', mean_val_disc_dice2, (epoch_num + 1) // save_interval)
            writer.add_scalar('Base2_Cup_Dice', mean_val_cup_dice2, (epoch_num + 1) // save_interval)
            
            mean_val_disc_dice3, mean_val_cup_dice3 = inference_riga(args=args, epoch=epoch_num, snapshot_path=snapshot_path, test_loader=valid_loader[2], model=model, test_save_path=None)
            writer.add_scalar('Base3_Disc_Dice', mean_val_disc_dice3, (epoch_num + 1) // save_interval)
            writer.add_scalar('Base3_Cup_Dice', mean_val_cup_dice3, (epoch_num + 1) // save_interval)

            # Original implementation's average calculations
            avg_disc_dice = (mean_val_disc_dice1 + mean_val_disc_dice2 + mean_val_disc_dice3) / 3
            avg_cup_dice = (mean_val_cup_dice1 + mean_val_cup_dice2 + mean_val_cup_dice3) / 3

            # Print original format scores
            logging.info(f'Current Average Validation Dice Scores - Disc: {avg_disc_dice:.4f}, Cup: {avg_cup_dice:.4f}')

            # Calculate average of all 6 scores for best model tracking
            current_scores = {
                'disc1': mean_val_disc_dice1, 'cup1': mean_val_cup_dice1,
                'disc2': mean_val_disc_dice2, 'cup2': mean_val_cup_dice2,
                'disc3': mean_val_disc_dice3, 'cup3': mean_val_cup_dice3
            }
            current_avg = sum(current_scores.values()) / 6

            # Print current validation dice scores
            logging.info(f'Epoch {epoch_num} - Current Validation Dice Scores:')
            logging.info(f'Target 1 - Disc: {mean_val_disc_dice1:.4f}, Cup: {mean_val_cup_dice1:.4f}')
            logging.info(f'Target 2 - Disc: {mean_val_disc_dice2:.4f}, Cup: {mean_val_cup_dice2:.4f}')
            logging.info(f'Target 3 - Disc: {mean_val_disc_dice3:.4f}, Cup: {mean_val_cup_dice3:.4f}')
            logging.info(f'Average Score: {current_avg:.4f}')

            # Check if current model is the best
            if current_avg > best_avg_score:
                best_avg_score = current_avg
                best_epoch = epoch_num
                best_scores = current_scores.copy()
                
                # Save best model
                best_model_path = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                logging.info(f'New best model saved at epoch {epoch_num}!')
                
                # Print original format best scores
                best_avg_disc = (best_scores['disc1'] + best_scores['disc2'] + best_scores['disc3']) / 3
                best_avg_cup = (best_scores['cup1'] + best_scores['cup2'] + best_scores['cup3']) / 3
                logging.info(f'Best Model Scores (Epoch {best_epoch}):')
                logging.info(f'Average Validation Dice Scores - Disc: {best_avg_disc:.4f}, Cup: {best_avg_cup:.4f}')
                
                # Print detailed best scores
                logging.info(f'Detailed Best Model Scores:')
                logging.info(f'Target 1 - Disc: {best_scores["disc1"]:.4f}, Cup: {best_scores["cup1"]:.4f}')
                logging.info(f'Target 2 - Disc: {best_scores["disc2"]:.4f}, Cup: {best_scores["cup2"]:.4f}')
                logging.info(f'Target 3 - Disc: {best_scores["disc3"]:.4f}, Cup: {best_scores["cup3"]:.4f}')
                logging.info(f'Best Overall Average Score: {best_avg_score:.4f}')

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            #save final model
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save final model to {}".format(save_mode_path))
            
            # Print final summary with both formats
            logging.info(f'Training finished! Best model was at epoch {best_epoch}')
            
            # Original format final best scores
            final_best_avg_disc = (best_scores['disc1'] + best_scores['disc2'] + best_scores['disc3']) / 3
            final_best_avg_cup = (best_scores['cup1'] + best_scores['cup2'] + best_scores['cup3']) / 3
            logging.info(f'Best Model Scores (Epoch {best_epoch}):')
            logging.info(f'Average Validation Dice Scores - Disc: {final_best_avg_disc:.4f}, Cup: {final_best_avg_cup:.4f}')
            
            # Detailed final best scores
            logging.info(f'Detailed Best Model Final Scores:')
            logging.info(f'Target 1 - Disc: {best_scores["disc1"]:.4f}, Cup: {best_scores["cup1"]:.4f}')
            logging.info(f'Target 2 - Disc: {best_scores["disc2"]:.4f}, Cup: {best_scores["cup2"]:.4f}')
            logging.info(f'Target 3 - Disc: {best_scores["disc3"]:.4f}, Cup: {best_scores["cup3"]:.4f}')
            logging.info(f'Best Overall Average Score: {best_avg_score:.4f}')
            break

    writer.close()
    return "Training Finished!"