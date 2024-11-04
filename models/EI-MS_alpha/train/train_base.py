import torch

import utils.data_utils as data_utils
import pdb
from math import sqrt


def train_model(dataset_loaders, model, optimizer, stat_names, selection_stat,
                train_func, args, select_higher=True, draw_func=None):
    best_stat = float('-inf') if select_higher else float('inf')
    best_model_path = ''

    train_output = open('%s/train_stats.csv' % args.output_dir, 'w+', buffering=1)
    valid_output = open('%s/valid_stats.csv' % args.output_dir, 'w+', buffering=1)
    test_output = open('%s/test_stats.csv' % args.output_dir, 'w+', buffering=1)

    for file in [train_output, valid_output, test_output]:
        file.write(','.join(stat_names) + '\n')

    #loading the pre trained model to resume training from the last saved model:
    # if from scratch, change it!
    EPOCH=791
    PATH='%s/model_791' % (args.model_dir)
    LOSS=1290079.0894
    
    model.load_state_dict(torch.load(PATH),strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch2 = EPOCH
    loss2 = LOSS

    patience_counter=0

    for epoch in range(epoch2+1,args.num_epochs):
        if patience_counter == 100:
            break

        args.epoch_num = epoch

        train_stats = train_func(
            data_loader=dataset_loaders['train'],
            model=model,
            optimizer=optimizer,
            stat_names=stat_names,
            mode='train',
            args=args,
            write_path=None)
        with torch.no_grad():
            valid_stats = train_func(
                data_loader=dataset_loaders['valid'],
                model=model,
                optimizer=None,
                stat_names=stat_names,
                mode='valid',
                args=args,
                write_path='%s/valid_%d' % (args.result_dir, epoch))
        print(data_utils.dict_to_pstr(
            train_stats, header_str='%d Train:' % epoch, key_list=stat_names))
        print(data_utils.dict_to_pstr(
            valid_stats, header_str='%d Valid:' % epoch, key_list=stat_names))
        train_output.write(
            data_utils.dict_to_dstr(train_stats, stat_names) + '\n')
        valid_output.write(
            data_utils.dict_to_dstr(valid_stats, stat_names) + '\n')

        model_path = '%s/model_%d' % (args.model_dir, epoch)

        save_model = False
        if select_higher:
            if valid_stats[selection_stat] > best_stat:
                best_stat = valid_stats[selection_stat]
                best_model_path = model_path
                save_model = True
        else:
            improved = False
            if valid_stats[selection_stat] < best_stat:
                best_stat = valid_stats[selection_stat]
                improved = True
                patience_counter=0
		
                best_model_path = model_path
                save_model = True

        if not improved:
            patience_counter=patience_counter+1

        if epoch == 0 or epoch == args.num_epochs - 1:
            save_model = True

        if save_model:
            torch.save(model.state_dict(), model_path)
            print('Model saved to %s' % model_path)

    train_output.close()
    valid_output.close()

    if best_model_path != '':
        model.load_state_dict(torch.load(best_model_path))
        print('Loading model from %s' % best_model_path)
        torch.save(model.state_dict(), '%s/model_best' % args.model_dir)

    with torch.no_grad():
        test_stats = train_func(
            data_loader=dataset_loaders['test'],
            model=model,
            optimizer=None,
            stat_names=stat_names,
            mode='test',
            args=args,
            write_path='%s/test_results' % args.result_dir)

        if draw_func is not None:
            draw_func(
                data_loader=dataset_loaders['test'],
                model=model,
                output_dir='%s/vis_output' % args.output_dir,
                args=args,)
    print(data_utils.dict_to_pstr(test_stats, header_str='Test:'))
    test_output.write(data_utils.dict_to_dstr(test_stats, stat_names) + '\n')
    test_output.close()

    return test_stats, best_model_path


def test_model(dataset_loaders, model, stat_names, train_func, args,
               inference_func=None):
    test_model_path = args.test_model
    print('Testing model loaded from %s' % test_model_path)
    model.load_state_dict(torch.load(test_model_path))

    with torch.no_grad():
        test_stats = train_func(
            data_loader=dataset_loaders['test'],
            model=model,
            optimizer=None,
            stat_names=stat_names,
            mode='test',
            args=args,
            write_path='%s/test.txt' % args.output_dir)
    print(data_utils.dict_to_pstr(test_stats, header_str='Test:'))
