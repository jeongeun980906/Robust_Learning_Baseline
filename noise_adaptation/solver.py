import torch

def pretrain(train_loader, model, optimizer,criterion, device='cuda'):
    model.train()
    avg_loss = 0
    for image,target,_,_ in train_loader:
        y = model(image.to(device))
        loss = criterion(y,target.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss+=loss.detach().cpu().item()
    return avg_loss/len(train_loader)

def train_adaptation(train_loader, model, adaptation_layer, optimizer
                ,noise_optimizer,criterion,device='cuda',weight_decay=1,beta=0.8):
    model.train()
    adaptation_layer.train()
    avg_loss = 0
    for image,target, _, _ in train_loader:
        try:
            h,y = model(image.to(device),returnh=True)
        except:
            h = model.conv1(image.to(device))
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)

            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
            h = model.layer4(h)

            h = model.avgpool(h)
            h = torch.flatten(h, 1)
            y = model.fc(h)
        z = adaptation_layer(h) # [N x D x D]
        y = torch.softmax(y,dim=-1)
        noise_pred = torch.matmul(y.unsqueeze(1),z).squeeze(1) # [N x D]
        weight_l2_norm = torch.norm(adaptation_layer.u)*weight_decay
        # print(weight_l2_norm)
        cross_entropy = criterion(noise_pred,target.to(device))
        soft_loss = torch.max(noise_pred,dim=-1)[0]
        soft_loss = torch.mean(soft_loss)
        # print(soft_loss,cross_entropy)
        loss = beta*cross_entropy+ (1-beta)*soft_loss+weight_l2_norm
        # torch.nn.utils.clip_grad_norm_(adaptation_layer.parameters(), 1)
        optimizer.zero_grad()
        noise_optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        noise_optimizer.step()
        avg_loss+=loss.detach().cpu().item()
    return avg_loss/len(train_loader)

def test_train(test_loader,model,device='cuda'):
    acc = 0
    total=0
    model.eval()
    for image,target,_,_ in test_loader:
        pred = model(image.to(device))
        pred = torch.argmax(pred,dim=-1)
        temp = (pred==target.to(device)).sum().cpu().item()
        total += pred.shape[0]
        acc += temp
    return acc/total

def test(test_loader,model,device='cuda'):
    acc = 0
    total=0
    model.eval()
    for image,target,_ in test_loader:
        pred = model(image.to(device))
        pred = torch.argmax(pred,dim=-1)
        temp = (pred==target.to(device)).sum().cpu().item()
        total += pred.shape[0]
        acc += temp
    return acc/total
