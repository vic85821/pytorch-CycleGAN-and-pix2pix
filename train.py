import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
#from util.visualizer import Visualizer
import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    #visualizer = Visualizer(opt)
    writer = SummaryWriter(os.path.join(opt.checkpoints_dir,'runs'))
    writer.add_text('opt', str(opt), 0)
    dummy_input = torch.rand(1, 3, 360, 360)
    writer.add_graph(model.netG, (dummy_input, dummy_input), False)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            #visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                visuals = model.get_current_visuals()
                images = []
                for label, image in visuals.items():
                    images.append(torch.squeeze(image.data))
                writer.add_image('images', vutils.make_grid(images[:8], normalize=True, nrow = 4), total_steps)
                writer.add_image('makeups', vutils.make_grid(images[8:], normalize=True, nrow = 4), total_steps)
                #save_result = total_steps % opt.update_html_freq == 0
                #visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            
            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                for k, v in losses.items():
                    writer.add_scalar('%s' % k, v, total_steps)
                #visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                #if opt.display_id > 0:
                    #visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
