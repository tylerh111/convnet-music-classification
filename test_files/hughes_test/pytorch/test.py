

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()

# switch to train mode
model.train()

end = time.time()
for i,(input,target) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    #target = target.cuda(async = True)
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # compute output
    #print('\t==>>input_var =',input_var)
    output = model(input_var)
    loss = criterion(output,target_var)

    # measure accuracy and record loss
    prec1,prec5 = accuracy(output.data,target,topk = (1,5))
    losses.update(loss.data[0],input.size(0))
    top1.update(prec1[0],input.size(0))
    top5.update(prec5[0],input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % print_freq == 0:
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch,i,len(train_loader),batch_time = batch_time,
            data_time = data_time,loss = losses,top1 = top1,top5 = top5))
