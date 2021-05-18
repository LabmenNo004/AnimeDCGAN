from tqdm import tqdm
import torch
import torchvision as tv
import torch.nn as nn
from torch.utils.data import DataLoader
import Load as Dataset


# config defines hyperparameters
class Config(object):

    # parameters
    data_path = 'D:\\files\\course\\DL\\Project\\Anime_Face\\data\\anim_face\\'
    virs = "result"
    num_workers = 8  
    img_size = 96  
    batch_size = 256 
    max_epoch = 100   
    lr1 = 2e-4  # generator learning rate
    lr2 = 2e-4  # D learning rate
    beta1 = 0.5  # Adam parameter
    gpu = True  
    nz = 100  # noise dimention
    ngf = 64  # Number of gen filters 
    ndf = 64  # Number of discriminator filters

    save_path = './data/imgs2' 

    d_every = 1  # train discriminator every batch
    g_every = 5  # train discriminator every 5 batches
    save_every = 10  # save model every 10 epochs
    netd_path = "./data/imgs2/netd.pth"
    netg_path = "./data/imgs2/netg.pth"

    # testing data
    gen_img = "result.png"
    # save 64 imgs each time
    gen_num = 64
    gen_search_num = 512
    # noise parameters
    gen_mean = 0 
    gen_std = 1 

conf = Config()

# Define generator
class GNet(nn.Module):
    def __init__(self, conf):
        super(GNet, self).__init__()
        self.ngf = conf.ngf
        self.Gene = nn.Sequential(
            # input dimension = conf.nz*1*1
            # output = (input - 1)*stride + output_padding - 2*padding + kernel_size
            nn.ConvTranspose2d(in_channels=conf.nz, out_channels=self.ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),

            # input: 4*4*ngf*8
            nn.ConvTranspose2d(in_channels=self.ngf * 8, out_channels=self.ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),

            # input: 8*8*ngf*4
            nn.ConvTranspose2d(in_channels=self.ngf * 4, out_channels=self.ngf * 2, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),

            # input: 16*16*ngf*2
            nn.ConvTranspose2d(in_channels=self.ngf * 2, out_channels=self.ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),

            # input: 32*32*ngf
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=3, kernel_size=5, stride=3, padding=1, bias=False),

            nn.Tanh(),

        )# output: 96*96*3

    def forward(self, x):
        return self.Gene(x)

# define discriminator
class DNet(nn.Module):
    def __init__(self, conf):
        super(DNet, self).__init__()

        self.ndf = conf.ndf
        self.Discrim = nn.Sequential(
            # input:(bitch_size, 3, 96, 96)
            # output:(bitch_size, ndf, 32, 32), (96 - 5 +2 *1)/3 + 1 =32
            nn.Conv2d(in_channels=3, out_channels=self.ndf, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace= True),

            # input:(ndf, 32, 32)
            nn.Conv2d(in_channels=self.ndf, out_channels= self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, True),

            # input:(ndf *2, 16, 16)
            nn.Conv2d(in_channels=self.ndf * 2, out_channels=self.ndf *4, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, True),

            # input:(ndf *4, 8, 8)
            nn.Conv2d(in_channels=self.ndf *4, out_channels= self.ndf *8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf *8),
            nn.LeakyReLU(0.2, True),

            # input:(ndf *8, 4, 4)
            # output:(1, 1, 1)
            nn.Conv2d(in_channels=self.ndf *8, out_channels=1, kernel_size=4, stride=1, padding= 0, bias=True),

            # using sigmoid for binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.Discrim(x).view(-1)


def train(**kwargs):

    for k_, v_ in kwargs.items():
        setattr(conf, k_, v_)

    # assign device
    if conf.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    # preprocessing
    transforms = tv.transforms.Compose([
        # 3*96*96
        tv.transforms.Resize(conf.img_size),   # resize to img_size * img_size
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    print(conf.data_path)
    dataset = tv.datasets.ImageFolder(root=conf.data_path, transform=transforms)

    dataloader = DataLoader(
        dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=conf.num_workers,
        drop_last=True
    )

    # initialize networks
    netg, netd = GNet(conf), DNet(conf)

    map_location = lambda storage, loc: storage


    # using torch.load to load model
    if conf.netg_path:
        print("load generator net")
        netg.load_state_dict(torch.load(f=conf.netg_path, map_location=map_location))
    if conf.netd_path:
        print("load discriminator net")
        netd.load_state_dict(torch.load(f=conf.netd_path, map_location=map_location))

    # move the model to GPU
    netd.to(device)
    netg.to(device)

    # using Adam optimizer
    optimize_g = torch.optim.Adam(netg.parameters(), lr=conf.lr1, betas=(conf.beta1, 0.999))
    optimize_d = torch.optim.Adam(netd.parameters(), lr=conf.lr2, betas=(conf.beta1, 0.999))

    # Using binary cross entropy loss
    criterions = nn.BCELoss().to(device)

    true_labels = torch.ones(conf.batch_size).to(device)
    fake_labels = torch.zeros(conf.batch_size).to(device)

    # generating random noises
    noises = torch.randn(conf.batch_size, conf.nz, 1, 1).to(device)

    # fixed noises to show test images
    fix_noises = torch.randn(conf.batch_size, conf.nz, 1, 1).to(device)

    # Training
    for epoch in range(conf.max_epoch):
        for ii_, (img, _) in tqdm((enumerate(dataloader))):
            real_img = img.to(device)

            # Training discriminator every batch
            if ii_ % conf.d_every == 0:
                optimize_d.zero_grad()

                # True images
                output = netd(real_img)
                # calculating loss
                loss_d_real = criterions(output, true_labels)
                loss_d_real.backward()

                # fake images
                noises = noises.detach()
                # generate images from noises
                fake_image = netg(noises).detach()
                # input the images into discriminator
                output = netd(fake_image)
                # calculating loss
                loss_d_fake = criterions(output, fake_labels)

                loss_d_fake.backward()

                # update parameter once for both true and fake images
                optimize_d.step()

            # Training generator every 5 batches
            if ii_ % conf.g_every == 0:
                optimize_g.zero_grad()
                # using different noises from the noise when training discriminator.
                noises.data.copy_(torch.randn(conf.batch_size, conf.nz, 1, 1))
                fake_image = netg(noises)
                output = netd(fake_image)
                # using true labels for loss calculation
                loss_g = criterions(output, true_labels)
                loss_g.backward()

                # update parameter
                optimize_g.step()

        # saving model
        if (epoch + 1) % conf.save_every == 0:
            fix_fake_image = netg(fix_noises)
            tv.utils.save_image(fix_fake_image.data[:64], "%s/%s.png" % (conf.save_path, epoch), normalize=True)

            torch.save(netd.state_dict(),  './data/imgs2/' + 'netd.pth')
            torch.save(netg.state_dict(),  './data/imgs2/' + 'netg.pth')



# does not calculate gradient
@torch.no_grad()
def generate(**kwargs):
    # generate images using trained model

    for k_, v_ in kwargs.items():
        setattr(conf, k_, v_)

    device = torch.device("cuda") if conf.gpu else torch.device("cpu")

    netg, netd = GNet(conf).eval(), DNet(conf).eval()

    map_location = lambda storage, loc: storage

    netd.load_state_dict(torch.load('./data/imgs2/netd.pth', map_location=map_location), False)
    netg.load_state_dict(torch.load('./data/imgs2/netg.pth', map_location=map_location), False)
    netd.to(device)
    netg.to(device)

    # generate images
    noise = torch.randn(conf.gen_search_num, conf.nz, 1, 1).normal_(conf.gen_mean, conf.gen_std).to(device)

    fake_image = netg(noise)
    score = netd(fake_image).detach()

    # selecting best image
    indexs = score.topk(conf.gen_num)[1]

    result = []

    for ii in indexs:
        result.append(fake_image.data[ii])

    # saving the image
    tv.utils.save_image(torch.stack(result), conf.gen_img, normalize=True, range=(-1, 1))

# does not calculate gradient
@torch.no_grad()
def generate_one_face(**kwargs):
# generate images using trained model
    for k_, v_ in kwargs.items():
        setattr(conf, k_, v_)

    device = torch.device("cuda") if conf.gpu else torch.device("cpu")

    netg, netd = GNet(conf).eval(), DNet(conf).eval()
    map_location = lambda storage, loc: storage

    netd.load_state_dict(torch.load('./data/imgs2/netd.pth', map_location=map_location), False)
    netg.load_state_dict(torch.load('./data/imgs2/netg.pth', map_location=map_location), False)
    netd.to(device)
    netg.to(device)

    noise = torch.randn(conf.gen_search_num, conf.nz, 1, 1).normal_(conf.gen_mean, conf.gen_std).to(device)

    fake_image = netg(noise)
    score = netd(fake_image).detach()

    # selecting best image
    indexs = score.topk(conf.gen_num)[1]

    i = 0
    for ii in indexs:
        result = []
        result.append(fake_image.data[ii])
        path = str(i) + ".jpg"
        tv.utils.save_image([fake_image.data[ii]], path, normalize=True, range=(-1, 1))
        i = i+1
        
    # saving the image
    tv.utils.save_image(torch.stack(result), conf.gen_img, normalize=True, range=(-1, 1))

def main():
    train()
    generate()


if __name__ == '__main__':
    generate_one_face()















