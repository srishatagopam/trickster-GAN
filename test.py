from time import time
from torch import optim
from collections import defaultdict
from dataloader import *
from models import *


def test(G, model, test_dir='./data/test/', lim=None):
    test = ImageDataset(test_dir, lim) if lim is not None else ImageDataset(test_dir)
    
    loader = DataLoader(test, batch_size=32, shuffle=True)
    G.to('cuda')
    model.to('cuda')
    model.eval()
    G.eval()
    test_dict = defaultdict(int, {i:0 for i in range(8)})
    perclass_correct = [0]*8
    perclass_totals = [0]*8
    correct = 0
    total = len(test)

    with torch.no_grad():
          for batch, (image, label) in enumerate(loader):
                image = image.to('cuda')
                pert = Resize(128)(G(image))
                pert = torch.clamp(pert, 0, 1)
                label = label.cpu().numpy()
                out = model(torch.clamp(image + pert, 0, 1))
                _, preds = torch.max(out, 1)
                preds = preds.detach().cpu().numpy()
                for i,j in zip(preds, label):
                    perclass_totals[j] += 1
                    if i==j:
                        perclass_correct[i] += 1
                correct += np.sum(np.equal(preds, label))
    print(f'Total Acc: {100 * correct / total}%')
    print('Per class Acc:')
    for idx, i in enumerate(test.keys):
        print(f'{test.classes[i]}: {100 * perclass_correct[idx] / perclass_totals[idx]}')
     